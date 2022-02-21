import random

random.seed(20222022)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    memory_dict = {
        "training":{
            "Cyclegan":{
                "x":[128, 256, 512], "y":[2939, 5517, 16357]
            },
            "CUT":{
                "x":[128, 256, 512], "y":[2365, 4133, 11123]
            },
            "F-LSeSim":{
                "x":[256, 512, 1024], "y":[3585, 8813, 23023]
            },
            "L-LSeSim":{
                "x":[256, 512, 1024], "y":[3605, 9331, 29289]
            }
        },
        "inference":{
            "Cyclegan":{
                "x":[128, 256, 512, 1024, 2048], "y":[1697, 1795, 2211, 3749, 12073]
            },
            "CUT":{
                "x":[128, 256, 512, 1024, 2048], "y":[1602, 1695, 2111, 3654, 11987]
            },
            "F-LSeSim":{
                "x":[512, 1024], "y":[8571, 19269]
            },
            "L-LSeSim":{
                "x":[512, 1024], "y":[8571, 19269]
            },
            "Cyclegan+KIN (ours)":{
                "x":[128, 256, 512, 1024, 2048, 4096, 9192], "y":[2307, 2307, 2307, 2307, 2307, 2307, 2307]
            },
            "CUT+KIN (ours)":{
                "x":[128, 256, 512, 1024, 2048, 4096, 9192], "y":[2307, 2307, 2307, 2307, 2307, 2307, 2307]
            },
            "F/L-LSeSim+KIN (ours)":{
                "x":[128, 256, 512, 1024, 2048, 4096, 9192], "y":[8581, 8581, 8581, 8581, 8581, 8581, 8581]
            }
        }
    }

    pal = sns.color_palette("husl", 8)
    hex_colors = list(map(matplotlib.colors.rgb2hex, pal))
    pal

    marker_list = ['o', 'X','v', 's', '^','p', 'D']
    plt.figure(figsize=(9,4))
    for idx, model_name in enumerate(memory_dict['training']):
        x = memory_dict['training'][model_name]["x"]
        y = memory_dict['training'][model_name]["y"]
        p2 = np.poly1d(np.polyfit(x, y, 2))
        
        xp = np.linspace(128, x[-1], 100)
        xp_extra = np.linspace(x[-1], 2000, 100)
        marker_on = []
        for x_ in x:
            marker_on.append(np.searchsorted(xp, x_, side='left'))
        # add rnd to avoid overlapping
        rnd = random.randint(0, 200)
        
        plt.plot(xp, p2(xp) + rnd, color=hex_colors[idx], linewidth=3.0,linestyle='-',markersize=8,marker=marker_list[idx],markevery=marker_on,label=model_name)
        plt.plot(xp_extra, p2(xp_extra) + rnd, color=hex_colors[idx], linewidth=3.0,linestyle='--')
        
    plt.xticks([128, 256, 512, 1024, 2048, 4096], [128, 256, 512, 1024, 2048, 4096], rotation=45)
    plt.yticks([5000, 10000, 15000, 20000, 25000, 30000], [5, 10, 15, 20, 25, 30])
    plt.ylim(0,32000)
    plt.title("Training")
    plt.ylabel("GPU Memory (GB)")
    plt.xlabel("Resolution (√x)")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("./training_usage.png", bbox_inches = 'tight')

    plt.figure(figsize=(9,4))
    for idx, model_name in enumerate(memory_dict['inference']):
        x = memory_dict['inference'][model_name]["x"]
        y = memory_dict['inference'][model_name]["y"]
        if not "KIN" in model_name:
            p2 = np.poly1d(np.polyfit(x, y, 2))
        else:
            p2 = np.poly1d(np.polyfit(x, y, 0))

        xp = np.linspace(128, x[-1], 100)
        xp_extra = np.linspace(x[-1], 10000, 100)
        marker_on = []
        
        # add rnd to avoid overlapping
        rnd = random.randint(0, 2000)
        
        for x_ in x:
            marker_on.append(np.searchsorted(xp, x_, side='left'))
        plt.plot(xp, p2(xp) + rnd, color=hex_colors[idx], linewidth=3.0,linestyle='-',markersize=8, marker=marker_list[idx],markevery=marker_on,label=model_name)
        if "KIN" in model_name:
            plt.plot(xp_extra, p2(xp_extra) + rnd, color=hex_colors[idx], linewidth=3.0,linestyle='-')
        else:
            plt.plot(xp_extra, p2(xp_extra) + rnd, color=hex_colors[idx], linewidth=3.0,linestyle='--')
        
    plt.xticks([256, 512, 1024, 2048, 4096, 9192], [256, 512, 1024, 2048, 4096, 9192], rotation=45)
    plt.yticks([5000, 10000, 15000, 20000, 25000, 30000], [5, 10, 15, 20, 25, 30])
    plt.ylim(0,32000)
    plt.ylabel("GPU Memory (GB)")
    plt.xlabel("Resolution (√x)")
    plt.title("Inference")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("./inference_usage.png", bbox_inches = 'tight')
