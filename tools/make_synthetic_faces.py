#!/usr/bin/env python3
"""Render and evaluate synthetic face illustrations for the MediBot demo.

Rendering uses Pillow and NumPy only. OpenCV is used by the command-line
evaluation to run the project's existing YuNet and SFace models.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
FACES = ROOT / "controllers" / "medibot_controller" / "faces"
MODELS = ROOT / "controllers" / "medibot_controller" / "models"
SIZE = 512


def _rgb(seed: int, salt: int, low: int, high: int) -> int:
    return int(np.random.default_rng(seed * 7919 + salt).integers(low, high))


def make_face(seed: int, *, variant: bool = False) -> Image.Image:
    """Draw one deterministic, softly rendered cartoon-realistic face."""
    rng = np.random.default_rng(seed)
    shift = 4.0 if variant else 0.0
    light = 1.035 if variant else 1.0
    bg = Image.new("RGB", (SIZE, SIZE), (239, 239, 235))
    # Supersampling makes curved feature edges smooth before final soft focus.
    s = 2
    im = bg.resize((SIZE*s, SIZE*s))
    d = ImageDraw.Draw(im, "RGBA")
    def xy(box): return tuple(int(round(v*s)) for v in box)
    cx = SIZE/2 + shift
    fw = float(rng.uniform(142, 174)); fh = float(rng.uniform(190, 218))
    chin = float(rng.uniform(.58, .82))
    skin = np.array([_rgb(seed, 1, 166, 229), _rgb(seed, 2, 112, 184), _rgb(seed, 3, 80, 151)], float)
    # Radial skin illumination and subtle texture are generated numerically.
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rx, ry = fw*.94, fh*1.05
    rad = ((xx-cx)/rx)**2 + ((yy-260)/ry)**2
    shade = np.clip(1.085 - .23*rad, .76, 1.08) * light
    noise = rng.normal(0, 1.35, (SIZE, SIZE, 1))
    skin_img = np.clip(skin[None,None,:]*shade[:,:,None] + noise, 0, 255).astype(np.uint8)
    skin_layer = Image.fromarray(skin_img, "RGB").resize((SIZE*s, SIZE*s), Image.Resampling.BICUBIC)
    # Face silhouette: broad temples tapering through cheeks into a rounded chin.
    face_pts = [(cx-fw*.76,155),(cx-fw*.91,205),(cx-fw*.88,292),(cx-fw*.69,365),
                (cx-fw*.43,414),(cx-fw*.22,440*chin+260*(1-chin)),(cx,448*chin+260*(1-chin)),
                (cx+fw*.22,440*chin+260*(1-chin)),(cx+fw*.43,414),(cx+fw*.69,365),
                (cx+fw*.88,292),(cx+fw*.91,205),(cx+fw*.76,155)]
    mask = Image.new("L", im.size, 0); md = ImageDraw.Draw(mask)
    md.polygon([(int(x*s),int(y*s)) for x,y in face_pts], fill=255)
    # Soft cast shadow behind head and neck.
    shadow = Image.new("RGBA", im.size, (0,0,0,0)); sd=ImageDraw.Draw(shadow)
    sd.ellipse(xy((cx-fw*.97,132,cx+fw*.97,444)), fill=(40,34,31,32))
    shadow=shadow.filter(ImageFilter.GaussianBlur(14*s)); im=Image.alpha_composite(im.convert("RGBA"),shadow)
    face_rgba = skin_layer.convert("RGBA"); face_rgba.putalpha(mask)
    im=Image.alpha_composite(im, face_rgba); d=ImageDraw.Draw(im,"RGBA")
    # Ears and neck
    d.ellipse(xy((cx-fw*.96,250,cx-fw*.72,325)), fill=tuple(skin.astype(int))+(255,))
    d.ellipse(xy((cx+fw*.72,250,cx+fw*.96,325)), fill=tuple(skin.astype(int))+(255,))
    d.rounded_rectangle(xy((cx-40,385,cx+40,475)), radius=20*s, fill=tuple((skin*.9).astype(int))+(255,))
    hair_choices=["short","wavy","parted","curly"]
    hair_style=hair_choices[int(rng.integers(0,len(hair_choices)))]
    hair=np.array([_rgb(seed, 4, 22, 98),_rgb(seed, 5, 15, 65),_rgb(seed, 6, 10, 45)])
    hc=tuple(hair.astype(int))+(255,)
    # Hair cap is drawn behind brows, with style-dependent silhouette.
    d.pieslice(xy((cx-fw*.97,92,cx+fw*.97,290)),180,360,fill=hc)
    d.rectangle(xy((cx-fw*.95,183,cx+fw*.95,220)),fill=hc)
    if hair_style=="parted":
        d.polygon([tuple(int(v*s) for v in p) for p in [(cx,110),(cx+90,145),(cx+12,167)]],fill=tuple((hair*.72).astype(int))+(255,))
    elif hair_style=="wavy":
        for hx in np.linspace(cx-fw*.78,cx+fw*.78,7):
            d.arc(xy((hx-24,145,hx+28,236)),0,180,fill=tuple((hair*.7).astype(int))+(210,),width=3*s)
    elif hair_style=="curly":
        for hx in np.linspace(cx-fw*.83,cx+fw*.83,9):
            hy=158+int(rng.integers(-18,18)); d.ellipse(xy((hx-19,hy-17,hx+19,hy+18)),fill=hc)
    # eyes/eyebrows, mildly varied spacing and scale
    eye_y=float(rng.uniform(243,260)); spacing=float(rng.uniform(49,62)); er=float(rng.uniform(23,30))
    iris_palette=[(68,112,135),(93,68,45),(64,112,74),(114,84,65)]
    iris=iris_palette[int(rng.integers(0,4))]
    for sign in (-1,1):
        ex=cx+sign*spacing
        d.ellipse(xy((ex-er-5,eye_y-er*.55-4,ex+er+5,eye_y+er*.58+5)),fill=(252,247,239,255))
        d.ellipse(xy((ex-er*.63,eye_y-er*.72,ex+er*.63,eye_y+er*.72)),fill=iris+(255,))
        d.ellipse(xy((ex-er*.30,eye_y-er*.45,ex+er*.30,eye_y+er*.46)),fill=(22,24,26,255))
        d.ellipse(xy((ex-er*.22-4,eye_y-er*.45-5,ex-er*.22+4,eye_y-er*.45+3)),fill=(255,255,255,235))
        d.arc(xy((ex-er-5,eye_y-er*.85-4,ex+er+5,eye_y+er*.55)),185,355,fill=(55,39,34,255),width=3*s)
        brow_y=eye_y-er-18
        thick=float(rng.uniform(5,9)); slope=sign*float(rng.uniform(-5,3))
        d.line([tuple(int(v*s) for v in p) for p in [(ex-er-8,brow_y+slope),(ex,brow_y),(ex+er+8,brow_y-slope)]],fill=hc,width=int(thick*s))
    # Nose bridge, soft shadow and nostrils.
    d.line([tuple(int(v*s) for v in p) for p in [(cx+5,eye_y+17),(cx+13,316),(cx+2,329)]],fill=tuple((skin*.78).astype(int))+(115,),width=5*s)
    d.ellipse(xy((cx-29,316,cx+30,344)),fill=(92,58,48,28))
    d.arc(xy((cx-28,308,cx+29,346)),15,165,fill=tuple((skin*.65).astype(int))+(205,),width=3*s)
    d.ellipse(xy((cx-20,333,cx-9,338)),fill=(77,53,48,115)); d.ellipse(xy((cx+9,333,cx+20,338)),fill=(77,53,48,115))
    # Mouth with a natural cupid bow and slight variant expression shift.
    mouth_y=365 + (2 if variant else 0); smile=float(rng.uniform(-4,5))+(2 if variant else 0)
    d.line([tuple(int(v*s) for v in p) for p in [(cx-38,mouth_y),(cx-15,mouth_y-3),(cx,mouth_y+1),(cx+15,mouth_y-3),(cx+38,mouth_y)]],fill=(112,55,57,240),width=4*s)
    d.arc(xy((cx-39,mouth_y-6,cx+39,mouth_y+20+smile)),8,172,fill=(130,58,61,255),width=4*s)
    d.line([tuple(int(v*s) for v in p) for p in [(cx-30,mouth_y+9),(cx,mouth_y+12+smile),(cx+30,mouth_y+9)]],fill=(239,186,177,190),width=2*s)
    # Optional subtle beard texture.
    if rng.random()<.38:
        beard=(45,37,35,55)
        d.arc(xy((cx-fw*.53,326,cx+fw*.53,425)),5,175,fill=beard,width=5*s)
        for _ in range(150):
            bx=float(rng.uniform(cx-fw*.48,cx+fw*.48)); by=float(rng.uniform(365,411))
            if ((bx-cx)/(fw*.5))**2+((by-365)/54)**2<1 and rng.random()<.75:
                d.point((int(bx*s),int(by*s)),fill=(38,32,31,int(rng.integers(20,65))))
    im=im.convert("RGB").resize((SIZE,SIZE),Image.Resampling.LANCZOS)
    if variant:
        im=im.rotate(4.0, resample=Image.Resampling.BICUBIC, fillcolor=(239,239,235))
    im=im.filter(ImageFilter.GaussianBlur(0.65))
    arr=np.asarray(im,dtype=np.float32)
    arr=np.clip(arr+rng.normal(0,1.15,arr.shape),0,255).astype(np.uint8)
    return Image.fromarray(arr,"RGB")


def _evaluate():
    import cv2
    det_path=MODELS/"yuNet.onnx"; rec_path=MODELS/"sface.onnx"
    if not det_path.is_file() or not rec_path.is_file():
        raise RuntimeError("Existing YuNet/SFace models are required; no downloads are attempted.")
    detector=cv2.FaceDetectorYN.create(str(det_path),"",(SIZE,SIZE),0.6,0.3,5000)
    recognizer=cv2.FaceRecognizerSF.create(str(rec_path),"")
    def embed(image):
        bgr=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        detector.setInputSize((bgr.shape[1],bgr.shape[0])); _, faces=detector.detect(bgr)
        if faces is None or not len(faces): return None,0.0
        row=max(faces,key=lambda f:float(f[14])); conf=float(row[14])
        if conf<.7:return None,conf
        aligned=recognizer.alignCrop(bgr,row); feat=np.asarray(recognizer.feature(aligned),np.float32).ravel()
        return feat/np.linalg.norm(feat),conf
    best=None
    last_images=None
    # Each seed supplies three candidate identities; at most 500 seeds are tried.
    for seed in range(500):
        family1=make_face(seed); family2=make_face(seed,variant=True)
        visitor1=make_face(seed+10000); visitor2=make_face(seed+20000)
        last_images=(family1,family2,visitor1,visitor2)
        items=[embed(x) for x in (family1,family2,visitor1,visitor2)]
        if any(e is None for e,c in items):
            continue
        a,b,v1,v2=[x[0] for x in items]
        scores={"family_1_vs_family_2":float(a@b),"visitor_1_vs_family":float(v1@a),
                "visitor_2_vs_family":float(v2@a),"visitor_1_vs_visitor_2":float(v1@v2)}
        quality=min(scores["family_1_vs_family_2"]-.5,.25-scores["visitor_1_vs_family"],
                    .25-scores["visitor_2_vs_family"],.25-scores["visitor_1_vs_visitor_2"])
        if best is None or quality>best[0]: best=(quality,seed,items,scores,(family1,family2,visitor1,visitor2))
        if quality>=0:
            for im,name in zip((family1,family2,visitor1,visitor2),
                               ("family_1.jpg","family_2.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")):
                im.save(FACES/name,quality=94)
            print(f"WINNING SEED {seed}; YuNet confidences {[round(x[1],3) for x in items]}; scores {scores}")
            return True
    if best:
        _,seed,items,scores,images=best
        print(f"NO ACCEPTABLE SET in 500 seeds. Best seed {seed}; confidences {[round(x[1],3) for x in items]}; scores {scores}")
        print("Synthetic drawings weren’t recognised with the required separation; thresholds and detector settings were not changed.")
        # Keep the strongest face-detected candidate for a usable preview, while reporting failure honestly.
        for im,name in zip(images,("family_1.jpg","family_2.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")):
            im.save(FACES/name,quality=94)
    else:
        print("NO ACCEPTABLE SET in 500 seeds. No complete four-face set passed YuNet confidence >= 0.7.")
        print("Synthetic drawings weren’t recognised with the required detector confidence; thresholds and detector settings were not changed.")
        if last_images:
            for im,name in zip(last_images,("family_1.jpg","family_2.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")):
                im.save(FACES/name,quality=94)
    _save_preview()
    return False


def _save_preview():
    """Write a labelled three-identity contact sheet using the chosen files."""
    names=("family_1.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")
    sheet=Image.new("RGB",(3*320,360),(245,245,242))
    draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        path=FACES/name
        image=Image.open(path).convert("RGB") if path.is_file() else make_face(50+i)
        image.thumbnail((300,300),Image.Resampling.LANCZOS)
        sheet.paste(image,(i*320+10,10))
        draw.text((i*320+10,320),name,fill=(35,35,35))
    sheet.save(ROOT/"faces_preview.png")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args(); FACES.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        for seed,name in enumerate(("family_1.jpg","family_2.jpg","VISITOR_UNKNOWN_01.jpg","VISITOR_UNKNOWN_02.jpg")):
            make_face(13 if seed<2 else 10013+seed,variant=(seed==1)).save(FACES/name,quality=94)
        _save_preview()
        return 0
    return 0 if _evaluate() else 2

if __name__=="__main__": raise SystemExit(main())
