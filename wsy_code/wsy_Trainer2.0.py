"""
动态的将需要的功能模块“注册”到Trainer类中，而不需要去修改Trainer最原始的定义，实现可扩展的功能。
对Trainer类进行升级，使他能够具备插件化处理的功能。
"""
class Trainer(object):
    def __init__(self):
        #这里定义了一个插件队列的字典， 保存不同时机调用的插件序列
        self.plugin_queues = {
            
        } 
    # 调用插件   
    def call_plugin(self):
        pass
    # 注册插件
    def register_plugin(self, plugin):
        pass