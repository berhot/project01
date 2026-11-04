"""Strict, transparent toy-image silhouette fidelity scoring.

Used only for evaluating generated PNG images, NEVER rendering/inference.
The fitted circle/square/triangle templates are ideal reference masks
within the foreground's bounding rectangle. They make a 32x32
geometry score that catches blocky 'circles' and square 'triangles'
that a coarse fill-ratio classifier may mislabel.

These scores do not replace human visual inspection.
"""
from PIL import Image, ImageDraw
import numpy as np
import torch

from src.pixel_metrics import analyze_image

SHAPES=("circle","square","triangle")
MINIMUM_IOU={"circle":0.90,"square":0.92,"triangle":0.85}
MINIMUM_SEPARATION=0.06


def _iou(a,b):
    inter=int(np.logical_and(a,b).sum())
    union=int(np.logical_or(a,b).sum())
    return inter/union if union else 1.0


def _shape_templates(bbox):
    left,top,right,bottom=bbox
    mid=(left+right)/2
    out={}
    for shape in SHAPES:
        canvas=Image.new("1",(32,32),0)
        draw=ImageDraw.Draw(canvas)
        if shape=="circle":
            draw.ellipse((left,top,right,bottom),fill=1)
        elif shape=="square":
            draw.rectangle((left,top,right,bottom),fill=1)
        else:
            upward=((mid,top),(right,bottom),(left,bottom))
            downward=((left,top),(right,top),(mid,bottom))
            variants=[]
            for polygon in (upward,downward):
                im=Image.new("1",(32,32),0)
                ImageDraw.Draw(im).polygon(polygon,fill=1)
                variants.append(np.asarray(im,dtype=bool))
            out[shape]=variants
            continue
        out[shape]=[np.asarray(canvas,dtype=bool)]
    return out


def shape_fidelity(image,expected_shape):
    """Evaluate actual model RGB [-1,1], with no access to its prompt tokens."""
    if expected_shape not in SHAPES:
        raise ValueError("Unknown expected v2 shape")
    if not isinstance(image,torch.Tensor) or image.shape!=(3,32,32):
        raise ValueError("Expected [3,32,32] RGB tensor")
    if not torch.isfinite(image).all():
        raise ValueError("Nonfinite image")
    rgb=((image.detach().cpu().clamp(-1,1)+1)*127.5).permute(1,2,0)
    background=np.array([246,246,244],dtype=np.float32)
    rgb_np=rgb.numpy()
    mask=np.max(np.abs(rgb_np-background),axis=2)>45.9
    ys,xs=np.where(mask)
    base={
        "expected_shape":expected_shape,
        "foreground_pixels":int(mask.sum()),
        "ideal_mask_iou_by_shape":{shape:0.0 for shape in SHAPES},
        "expected_mask_iou":0.0,
        "shape_separation":0.0,
        "touches_image_border":True,
        "qualified_geometry":False,
    }
    if len(xs)<20:
        return base
    left,right=int(xs.min()),int(xs.max())
    top,bottom=int(ys.min()),int(ys.max())
    if min(right-left+1,bottom-top+1)<7:
        return base
    base["touches_image_border"]=bool(
        left==0 or top==0 or right==31 or bottom==31
    )
    templates=_shape_templates((left,top,right,bottom))
    ious={shape:round(max(_iou(mask,ideal) for ideal in variants),5)
          for shape,variants in templates.items()}
    best_other=max(v for name,v in ious.items() if name!=expected_shape)
    target=ious[expected_shape]
    margin=round(target-best_other,5)
    base.update({
        "bbox":[left,top,right,bottom],
        "ideal_mask_iou_by_shape":ious,
        "expected_mask_iou":target,
        "shape_separation":margin,
        "qualified_geometry":bool(
            target>=MINIMUM_IOU[expected_shape]
            and margin>=MINIMUM_SEPARATION
            and not base["touches_image_border"]
        ),
    })
    return base


def score_model_sample(image,expected_color,expected_shape):
    from src.pixel_dataset_v2 import COLORS
    if expected_color not in COLORS:
        raise ValueError("Unsupported target color")
    quality=shape_fidelity(image,expected_shape)
    prediction=analyze_image(image,dataset_version="v2")
    quality["color_correct"]=bool(prediction["color"]==expected_color)
    quality["legacy_predicted_shape"]=prediction["shape"]
    quality["strict_prompt_pass"]=bool(
        quality["color_correct"] and quality["qualified_geometry"]
    )
    return quality
