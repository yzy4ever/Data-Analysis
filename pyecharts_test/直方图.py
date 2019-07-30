#官方文档https://pyecharts.org
from pyecharts.charts import Bar   #柱状图
from pyecharts import options as opts
from pyecharts.globals import ThemeType   #主题
from pyecharts.options import ToolboxOpts   #工具条

bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK,width="1200px",height="700px"))

        .set_global_opts(
            title_opts={"text": "主标题", "subtext": "副标题"},
            toolbox_opts=ToolboxOpts(is_show=True,orient="horizontal"),   #工具栏
            datazoom_opts= [opts.DataZoomOpts(range_start=10, range_end=80, is_zoom_lock=False)]   #数据滑动条
            )

        .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
        .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])

        .render('./picture1.html')
)