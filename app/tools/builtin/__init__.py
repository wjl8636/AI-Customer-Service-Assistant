# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
# ch08 内置工具包:每个模块 = @tool 实现 + registry.register(...) 声明。
# registry.scan_builtin() 在服务启动(lifespan)时导入本包全部模块,import 即注册。
# 新增内置工具 = 往本包丢一个文件,核心代码零改动。
