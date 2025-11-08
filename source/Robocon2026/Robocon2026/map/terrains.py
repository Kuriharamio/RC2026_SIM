import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TERRAINS_CFG = TerrainGeneratorCfg(
    # 每个地形块大小为8x8米
    size=(8.0, 8.0),
    # 边界宽度20米
    border_width=20.0,
    # 地形网格为10行20列
    num_rows=10,
    num_cols=20,
    # 水平和垂直分辨率
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # 包含两种子地形
    sub_terrains={
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.1
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.4,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.3,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.8,
            grid_width=1.2,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, 
        #     noise_range=(0.02, 0.10), 
        #     noise_step=0.02, 
        #     border_width=0.25
        # ),
    },
)
