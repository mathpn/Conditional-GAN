"""
Generate GIF with sampled images during training.

Arguments:
    - 'file_prefix': prefix of the file names of the images.
"""

import sys
import glob

from PIL import Image
 

def main():
    prefix = sys.argv[1]
    frames = []
    imgs = glob.glob(f"{prefix}*.png")
    imgs = sorted(imgs, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    frames[0].save(f'{prefix}.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)
    print(f'{prefix}.gif saved.')


if __name__ == "__main__":
    main()
