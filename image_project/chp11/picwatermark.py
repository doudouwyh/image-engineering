"""
    watermark
"""

from PIL import Image, ImageDraw, ImageFont
import time


def watermark(img, img_ed, date, fnt):

    img_draw = ImageDraw.Draw(img_ed)
    img_draw.text((100, 100), date, font=fnt, fill=(255, 255, 255, 255))

    out = Image.composite(img, img_ed, img)
    out = out.resize((270, 270))
    out.save("watermark.gif")
    exit()


def watermark_test():
    img = Image.open("../pic/lena_color.jpg").convert('RGBA')

    date = time.strftime("%Y/%m/%d")

    img_ed = Image.new('RGBA', img.size, (255, 255, 255, 0))
    fnt = ImageFont.truetype("c:/Windows/fonts/REFSAN.TTF", 30)
    watermark(img, img_ed, date,fnt)


if __name__ == '__main__':
    watermark_test()