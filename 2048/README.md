# py2048game

> 基于PyGame开发的2048游戏。

本程序在 Windows 10 和 Python 3.12.2 的环境下编写。需要安装`PyGame`库才能运行。

用 Python 运行 `2048Game.py` 文件，游戏窗口随即显示。

什么是2048游戏？怎么玩？

- 百度百科: https://baike.baidu.com/item/2048/13383511
- Wikipedia: https://zh.wikipedia.org/wiki/2048_(%E9%81%8A%E6%88%B2)

# 修改代码

根据代码中的注释修改。下方列出的是比较常用的变量或常量。

```python
gameMap = [[0 for _ in range(4)] for _ in range(4)]  # 游戏地图。前者是宽度，后者是高度。可以随意更改
INTERVAL = 500  # 常量(毫秒): 新方块生成时等待一段时间，便于用户查看
SIDE = 90  # 常量(像素): 方块边长

screen = pygame.display.set_mode((410, 500))
```

后面的`getBestKeySimple()`是我随便写的，这个算法甚至没有赢过一场，可以删掉。

# 打包为exe文件

使用`PyInstaller`即可，先安装，然后运行`bulid.cmd`，等待运行完成后，打包好的程序在运行目录下的`dist`文件夹中。

# 运行图片

![](https://github.com/user-attachments/assets/04426200-4fca-4c50-98a4-25efdc34d67d)

# License

[MIT](https://github.com/yzl3014/py2048game/blob/main/LICENSE)

图标`icon.ico`来自网络。
