def auto_text_layout(inputext=[], x=0,y=0,row_spacing=100, col_spacing=80, vertical=True, font="SimSun", color=(255, 255, 255, 0)):
    if vertical:
        output=[] 
        y0=y
        for text in inputext:
            for char in list(text):
                output.append({
                    "content": char,
                    "position": (x, y),
                    "font": font,
                    "color": color
                })
                y +=   col_spacing
            y= y0
            x += row_spacing
    else:
        output=[]
        x0 = x
        for text in inputext:
            for char in list(text):
                output.append({
                    "content": char,
                    "position": (x, y),
                    "font": font,
                    "color": color
                })
                x += row_spacing
            x=x0
            y += col_spacing
    return output
text_lines = ['    悠闲的夏日，','   不在乎目的地，', '在乎的是沿途的风景', ' 以及看风景的心情.'] 
auto_text=auto_text_layout(inputext=text_lines, x=510,y=85,row_spacing=48, col_spacing=63, vertical=False, font="SimSun", color=(255, 255, 255, 0))
print(auto_text)
