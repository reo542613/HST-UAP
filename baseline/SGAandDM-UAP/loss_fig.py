# -*- coding: utf-8 -*-
"""
分别拼接 DM 与 SGA 损失曲线图（兼容 Pillow 10）
python stitch_two_sep.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

# 兼容 Pillow 10 与旧版
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCUBIC

models   = ["AlexNet", "VGG16", "DEITB", "SWINB", "VITB"]
name_map = {"DEITB": "DeiT-B", "SWINB": "Swin-B", "VITB": "ViT-B"}

def find_loss_png(model, root):
    """在 root/<model>/ 下找 *<model>*loss_epoch.png"""
    dir_path = os.path.join(root, model)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在：{dir_path}")
    for f in os.listdir(dir_path):
        if f.endswith("loss_epoch.png") and model in f:
            return os.path.join(dir_path, f)
    raise FileNotFoundError(f"未找到 *{model}*loss_epoch.png 于 {dir_path}")

def stitch_one_set(root, save_name):
    """拼接指定根目录下的图并保存"""
    imgs = [Image.open(find_loss_png(m, root)) for m in models]

    # 统一高度
    target_h = max(im.height for im in imgs)
    resized  = [im.resize((int(im.width * target_h / im.height), target_h), LANCZOS) for im in imgs]

    # 水平拼接 + 底部留白
    total_w = sum(im.width for im in resized)
    canvas  = Image.new("RGB", (total_w, target_h + 40), (255, 255, 255))

    x_offset = 0
    for im in resized:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width

    # 写模型名（居中，兼容 Pillow 10）
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("simhei.ttf", 28)
    except:
        font = ImageFont.load_default()

    x_offset = 0
    for i, m in enumerate(models):
        w   = resized[i].width
        txt = name_map.get(m, m)
        # 文字宽度兼容
        if hasattr(font, 'getbbox'):
            tw = font.getbbox(txt)[2] - font.getbbox(txt)[0]
        else:
            tw, _ = draw.textsize(txt, font=font)
        draw.text((x_offset + (w - tw) // 2, target_h + 5), txt, fill=(0, 0, 0), font=font)
        x_offset += w

    canvas.save(save_name, dpi=(300, 300))
    print(f"✅ 已保存 → {save_name}")

# 分别生成两张图
stitch_one_set("DM",   "dm_loss_curve.png")
stitch_one_set("SGA",  "sga_loss_curve.png")