from dashscope import MultiModalConversation
import dashscope
dashscope.api_key ='把你申请的qwenvl api-key放这边'
def call_with_local_file():
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """
    local_file_path1 = 'file:///Users/**/output_posters/poster_with_multiple_lines.jpg'
    local_file_path2 = 'file://The_local_absolute_file_path2'
    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': local_file_path1
            },
            {
                'text': '请描述这张图,这张图中文字放置的位置合理吗？符合审美需求吗？\
                        1.如果合理请回复是,并给出合理原因\
                        2.如果不合理给出理由和建议\
                        3.如果有建议请给出文字合适放置的坐标位置\
                        4.如果不合理给出字体大小建议\
                        5.如果不合理给出字体颜色建议\
                        6.json格式输出回答结果'
            },
        ]
    }]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages)
    print(response)


if __name__ == '__main__':
    call_with_local_file()

