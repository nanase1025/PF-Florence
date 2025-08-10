import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_image(image_path):
    # 加载图像
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


# def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = "cuda" if not cpu_only else "cpu"
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     _ = model.eval()
#     return model
def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameters (M): {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable parameters (M): {trainable_params/1e6:.2f}M")
    
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, cpu_only=False):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # 基于阈值过滤
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    
    # 获取短语
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit in logits_filt[filt_mask]:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def convert_to_xyxy(box, w, h):
    """
    将中心点+宽高格式 [center_x, center_y, width, height] 转换为左上右下坐标 [x1, y1, x2, y2]
    所有坐标都是归一化后的 (0-1之间)
    """
    cx, cy, bw, bh = box
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return [x1, y1, x2, y2]


def convert_to_florence_format(object_name, boxes):
    """
    将object_name和边界框转换为florence格式
    例如: couch<loc_449><loc_287><loc_923><loc_891>
    """
    if not boxes:
        return object_name
    
    florence_string = object_name
    for box in boxes:
        # 将归一化坐标转换为0-999范围
        x1, y1, x2, y2 = box
        loc_x1 = min(int(x1 * 1000), 999)
        loc_y1 = min(int(y1 * 1000), 999)
        loc_x2 = min(int(x2 * 1000), 999)
        loc_y2 = min(int(y2 * 1000), 999)
        
        # 确保坐标在0-999范围内
        loc_x1 = max(0, loc_x1)
        loc_y1 = max(0, loc_y1)
        loc_x2 = max(0, loc_x2)
        loc_y2 = max(0, loc_y2)
        
        # 添加位置标记
        florence_string += f"<loc_{loc_x1}><loc_{loc_y1}><loc_{loc_x2}><loc_{loc_y2}>"
    
    return florence_string


def process_json_file(json_file, images_dir, config_file, checkpoint_path, output_file, box_threshold=0.3, text_threshold=0.25, cpu_only=False, max_images=None):
    # 加载模型
    print("正在加载模型...")
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    
    # 读取JSON文件
    print(f"正在读取JSON文件: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 处理每个图像
    print("开始处理图像...")
    processed_count = 0
    
    # 如果指定了最大图片数量，创建一个仅包含需要处理的部分的新数据字典
    if max_images is not None:
        limited_data = {}
        for img_id, img_data in list(data.items())[:max_images]:
            limited_data[img_id] = img_data
        process_data = limited_data
        print(f"限制处理前 {max_images} 个图像")
    else:
        process_data = data
    
    for img_id, img_data in tqdm(process_data.items()):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像不存在 {img_path}, 跳过")
            continue
        
        # 加载图像
        image_pil, image = load_image(img_path)
        img_width, img_height = image_pil.size
        
        # 处理每个边界框
        for bbox_item in img_data.get("bboxes", []):
            object_name = bbox_item.get("object_name", "")
            
            if not object_name:
                print(f"警告: 图像 {img_id} 中的边界框缺少 object_name, 跳过")
                continue
            
            # 使用Grounding DINO查询对象
            boxes, phrases = get_grounding_output(
                model, image, object_name, box_threshold, text_threshold, cpu_only=cpu_only
            )
            
            # 将DINO输出添加到bbox中
            if len(boxes) > 0:
                # 转换为左上右下坐标格式（归一化坐标）
                xyxy_boxes = [convert_to_xyxy(box.tolist(), img_width, img_height) for box in boxes]
                
                # 添加原始DINO输出（左上右下坐标格式）
                bbox_item["dino_out"] = xyxy_boxes
                
                # 添加florence格式的输出字符串
                florence_string = convert_to_florence_format(object_name, xyxy_boxes)
                bbox_item["dino2florence_format"] = florence_string
            else:
                # 如果没有检测到任何边界框
                bbox_item["dino_out"] = []
                bbox_item["dino2florence_format"] = object_name  # 只有对象名称，没有位置标记
        
        processed_count += 1
        if max_images is not None and processed_count >= max_images:
            break
    
    # 如果只处理部分图片，将处理后的数据合并回原始数据
    if max_images is not None:
        for img_id, img_data in process_data.items():
            data[img_id] = img_data
    
    # 保存更新后的JSON文件
    print(f"保存结果到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"处理完成! 总共处理了 {processed_count} 个图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("处理JSON文件并添加Grounding DINO输出", add_help=True)
    parser.add_argument("--json_file", type=str, default="merged_affordances_flo_v1_internvl_test.json", help="输入JSON文件路径")
    parser.add_argument("--images_dir", type=str, default="data/sunrgbd_jpgs", help="图像目录路径")
    parser.add_argument("--config_file", type=str, default="groundingdino/config/GroundingDINO_SwinB_cfg.py", help="DINO配置文件路径")
    parser.add_argument("--checkpoint_path", type=str, default="dinockpt/groundingdino_swinb_cogcoor.pth", help="DINO检查点文件路径")
    parser.add_argument("--output_file", type=str, default="result/internvl_test_result.json", help="输出JSON文件路径")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="边界框阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="文本阈值")
    parser.add_argument("--cpu-only", action="store_true", help="仅使用CPU运行")
    parser.add_argument("--max_images", type=int, default=None, help="最多处理的图像数量，默认为处理所有图像")
    args = parser.parse_args()

    process_json_file(
        args.json_file,
        args.images_dir,
        args.config_file,
        args.checkpoint_path,
        args.output_file,
        args.box_threshold,
        args.text_threshold,
        args.cpu_only,
        args.max_images
    )