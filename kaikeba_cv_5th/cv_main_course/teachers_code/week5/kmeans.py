# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })


# %%
df


# %%
k = 3


# %%
centroids = {
       i: [np.random.randint(0, 80), np.random.randint(0, 80)]
       for i in range(k)
}


# %%
centroids


# %%
colormap = {0: 'r', 1: 'g', 2: 'b'}


# %%
def paint_color(df, centroids, colmap):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (   # 计算每个点和中心点之间的距离，分别放在df的一列中
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]  # 生成一个distance_from_i字符串列表
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)  # df.loc[:, distance_from_centroid_id] 区域选择选区列明是distance_from_i的列
    #  .idxmin(axis=1)选取每行最小值的索引，即“distance_from_i”
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))  # lstrip() 方法用于截掉字符串左边的空格或指定字符
    # 这里是找到最近的中心点，在closest这一列中只留下中心点的序号
    df['color'] = df['closest'].map(lambda x: colmap[x])  # 将中心点的序号转换成对应的颜色
    return df


# %%
def update(df, centroids):  # 更新中心点的位置
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])  # 计算第i个点最近的所有点x值的平均值
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])  # 计算第i个点最近的所有点y值的平均值
    return centroids


# %%
paint_color(df,centroids,colormap)


# %%
df_1 = paint_color(df,centroids,colormap)


# %%
cen_1 = update(df_1,centroids)


# %%
cen_1


# %%
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colormap[i], linewidths=6)
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()


# %%
for i in range(10):
        closest_centroids = df['closest'].copy(deep=True)  # 深复制
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colormap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = paint_color(df, centroids, colormap)

        if closest_centroids.equals(df['closest']):
            break


# %%


