import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_transforms():
    """
    Returns image augmentation pipeline designed for HTR without destroying 
    fine diacritics features. Resizing is handled dynamically in the Dataset class.
    """
    return A.Compose([
        # 1. Xoay/Nghiêng cực kỳ nhẹ (tránh xoay mạnh làm biến đổi hoặc mất dấu phụ)
        A.ShiftScaleRotate(
            shift_limit=0.03, 
            scale_limit=0.03, 
            rotate_limit=3, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=[255, 255, 255], 
            p=0.4
        ),
        
        # 2. Giả lập bóng mờ hoặc thay đổi độ sáng tối do chụp tài liệu bằng điện thoại
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.4
        ),
        
        # 3. Giả lập nét chữ mờ do rung tay hoặc nhòe mực nhẹ
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.25),
        
        # 4. Chuẩn hóa theo chuẩn ImageNet
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_transforms():
    """
    Returns evaluation transform (normalize and convert to tensor).
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

