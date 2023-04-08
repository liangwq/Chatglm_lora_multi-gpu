image = pipe(''.join(prompt),negative_prompt =negative_prompt , generator=generator,height = 512,width=768).images[0] 
image
## 新增sd 的 lora效果控制 ##
