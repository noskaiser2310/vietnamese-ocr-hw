import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
import random
from PIL import Image

class VietnameseHTRDataset(Dataset):
    """
    Dataset class for loading Vietnamese handwritten text images with annotations.
    """
    def __init__(self, annotation_file, img_dir, vocab_file, transform=None, max_target_len=90):
        """
        Args:
            annotation_file (str): Path to annotation txt file (format: path \t text).
            img_dir (str): Path to the folder containing handwriting images.
            vocab_file (str): Path to save/load the vocabulary json file.
            transform (callable, optional): Optional transform to be applied on an image.
            max_target_len (int): Maximum length of target character sequence.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.max_target_len = max_target_len
        
        # 1. Đọc danh sách annotations
        self.samples = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))
                elif len(parts) == 1 and parts[0]:
                    # Phòng trường hợp dòng text rỗng
                    self.samples.append((parts[0], ""))
                    
        # 2. Xây dựng hoặc load bộ từ điển Vocab
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
        self.vocab_file = vocab_file
        
        if os.path.exists(vocab_file):
            print(f"Loading vocabulary from {vocab_file}...")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                # Tương thích với cấu trúc của vocab.json tự sinh từ EDA
                if isinstance(vocab_data, dict) and "char_to_id" in vocab_data:
                    self.char2idx = vocab_data["char_to_id"]
                else:
                    self.char2idx = vocab_data
        else:
            print("Building vocabulary from annotations...")
            unique_chars = set()
            for _, text in self.samples:
                unique_chars.update(list(text))
            
            # Sắp xếp các ký tự độc nhất để đảm bảo tính nhất quán
            sorted_chars = sorted(list(unique_chars))
            
            # Tạo mapping
            self.char2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
            for char in sorted_chars:
                if char not in self.char2idx:
                    self.char2idx[char] = len(self.char2idx)
                    
            # Lưu vocab ra file json
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(self.char2idx, f, ensure_ascii=False, indent=4)
            print(f"Vocabulary built with {len(self.char2idx)} tokens and saved to {vocab_file}.")
            
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.pad_token_id = self.char2idx["<pad>"]
        self.sos_token_id = self.char2idx["<s>"]
        self.eos_token_id = self.char2idx["</s>"]
        self.unk_token_id = self.char2idx["<unk>"]

        # 3. Tạo hoặc tải Cache kích thước ảnh phục vụ Bucketing cực nhanh
        cache_file = annotation_file.replace(".txt", "_metadata.json")
        if os.path.exists(cache_file):
            print(f"Loading image metadata cache from {cache_file}...")
            with open(cache_file, "r") as f_cache:
                self.metadata = json.load(f_cache)
        else:
            print("Scanning image dimensions and building metadata cache...")
            self.metadata = {}
            for img_rel_path, text in self.samples:
                img_path = os.path.join(self.img_dir, os.path.basename(img_rel_path))
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            self.metadata[img_rel_path] = {"w": w, "h": h, "ratio": w / h}
                    except Exception:
                        self.metadata[img_rel_path] = {"w": 700, "h": 64, "ratio": 11.0}
                else:
                    self.metadata[img_rel_path] = {"w": 700, "h": 64, "ratio": 11.0}
            
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f_cache:
                    json.dump(self.metadata, f_cache)
                print(f"Saved metadata cache to {cache_file}.")
            except Exception as e:
                print(f"Warning: Could not save metadata cache: {e}")
        
    def __len__(self):
        return len(self.samples)
        
    def text_to_ids(self, text):
        """
        Chuyển đổi chuỗi text thành list of token IDs kèm <s> và </s>.
        """
        ids = [self.sos_token_id]
        for char in text:
            ids.append(self.char2idx.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        
        # Cắt bớt nếu vượt quá độ dài tối đa
        if len(ids) > self.max_target_len:
            ids = ids[:self.max_target_len - 1] + [self.eos_token_id]
            
        return ids

    def ids_to_text(self, ids):
        """
        Chuyển đổi list of token IDs ngược lại thành chuỗi text (Greedy Decode helper).
        """
        chars = []
        for idx in ids:
            if idx in [self.sos_token_id, self.pad_token_id]:
                continue
            if idx == self.eos_token_id:
                break
            chars.append(self.idx2char.get(idx, "<unk>"))
        return "".join(chars)
        
    def __getitem__(self, idx):
        img_rel_path, text = self.samples[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(img_rel_path))
        
        # Load ảnh
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Nếu ảnh lỗi, trả về ảnh trắng tạm thời
            print(f"[WARNING] Error reading image {img_path}: {e}")
            image = Image.new("RGB", (700, 64), color="white")
            
        # 1. Resize ảnh giữ nguyên tỉ lệ aspect ratio về height chuẩn = 128
        # Điều này đảm bảo nét chữ và dấu phụ không bao giờ bị bóp méo
        w, h = image.size
        aspect_ratio = w / h
        new_h = 128
        new_w = int(new_h * aspect_ratio)
        
        # Giới hạn chiều rộng tối đa và tối thiểu tránh lỗi bộ nhớ hoặc dị biệt
        new_w = max(128, min(new_w, 2048))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Tokenize nhãn văn bản
        target_ids = self.text_to_ids(text)
        
        # Chuyển đổi định dạng ảnh bằng Albumentations
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            # Fallback đơn giản chuyển thành tensor
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return {
            "image": image_tensor,
            "target": torch.tensor(target_ids, dtype=torch.long),
            "target_len": len(target_ids)
        }
        
    def collate_fn(self, batch):
        """
        Collate function hỗ trợ Dynamic Padding (Pad ảnh và Text theo batch lớn nhất).
        Do ảnh được gom nhóm theo aspect ratio, max_w của batch sẽ rất gần với w của các ảnh khác,
        giúp giảm thiểu tối đa padding vô nghĩa.
        """
        images = [item["image"] for item in batch]
        targets = [item["target"] for item in batch]
        target_lens = [item["target_len"] for item in batch]
        
        # 1. Pad ảnh về cùng size trong batch (Dynamic Padding)
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])
        
        padded_images = []
        for img in images:
            c, h, w = img.shape
            # Tạo tensor trắng tinh làm nền padding (Handwriting background là màu trắng)
            # Với ảnh đã chuẩn hóa Normalize ImageNet, giá trị 1.0 đến 2.0 tương ứng nền trắng
            padded_img = torch.ones((c, max_h, max_w), dtype=img.dtype)
            # Copy ảnh gốc vào góc trên bên trái
            padded_img[:, :h, :w] = img
            padded_images.append(padded_img)
            
        # 2. Pad chuỗi nhãn mục tiêu (targets)
        max_target_len = max(target_lens)
        padded_targets = []
        for tgt in targets:
            pad_size = max_target_len - len(tgt)
            if pad_size > 0:
                padded_tgt = torch.cat([tgt, torch.tensor([self.pad_token_id] * pad_size, dtype=torch.long)])
            else:
                padded_tgt = tgt
            padded_targets.append(padded_tgt)
            
        return {
            "images": torch.stack(padded_images),
            "targets": torch.stack(padded_targets),
            "target_lens": torch.tensor(target_lens, dtype=torch.long)
        }


class AspectGroupedBatchSampler(Sampler):
    """
    BatchSampler gom nhóm các mẫu có Aspect Ratio tương đương nhau vào chung một batch
    để tối ưu hóa bộ nhớ và triệt tiêu padding thừa thãi khi training.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Quét aspect ratio từ dataset cache
        self.indices_with_ratios = []
        for idx in range(len(dataset)):
            img_rel_path, _ = dataset.samples[idx]
            ratio = dataset.metadata.get(img_rel_path, {}).get("ratio", 11.0)
            self.indices_with_ratios.append((idx, ratio))
            
    def __iter__(self):
        # 1. Sắp xếp toàn bộ chỉ số theo aspect ratio tăng dần
        sorted_indices = sorted(self.indices_with_ratios, key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in sorted_indices]
        
        batches = []
        if self.shuffle:
            # 2. Tạo local shuffling bằng cách chia thành các blocks lớn
            # Tránh việc mô hình luôn luôn học ảnh ngắn trước ảnh dài
            block_size = self.batch_size * 100
            blocks = [sorted_indices[i:i + block_size] for i in range(0, len(sorted_indices), block_size)]
            
            for block in blocks:
                # Shuffle ngẫu nhiên trong phạm vi block cục bộ (độ giãn tỉ lệ gần tương đương)
                random.shuffle(block)
                for i in range(0, len(block), self.batch_size):
                    batch = block[i:i + self.batch_size]
                    if len(batch) == self.batch_size:
                        batches.append(batch)
        else:
            # Không shuffle (chạy Validation hoặc Test)
            for i in range(0, len(sorted_indices), self.batch_size):
                batch = sorted_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    
        # 3. Shuffle thứ tự của các batch
        if self.shuffle:
            random.shuffle(batches)
            
        return iter(batches)
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
