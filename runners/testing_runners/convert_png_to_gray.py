from os.path import join

from PIL import Image

images = [
    "/home/papalotl/Desktop/compartments/190301/017/017.png",
    "/home/papalotl/Desktop/compartments/190329/004/004.png",
    "/home/papalotl/Desktop/compartments/190329/007/007.png",
    # "/home/papalotl/Desktop/compartments/190329/010/010.png",
]
save_as_png = True
output_folder = "/scratch/trueba/yeast/mitochondria/test/raw/926pix"
type_f = "raw"
for image_path in images:
    tomo_number = image_path[-7:-4]
    png_name = image_path[-18:-12] + "_" + tomo_number + ".png"
    # if type_f == "raw":
    #     png_name = image_path[-14:]
    #     path_vol = join('volumes/raw', vol_name)
    # if type_f == "label":
    #     png_name = tomo_number + ".png"
    #     path_vol = join('volumes/labels/mito', vol_name)
    #
    # # print(path_vol)
    #
    if save_as_png:
        # png_name_grey = vol_name + ".png"
        # output_path = join(output_folder, png_name_grey)
        output_path = join(output_folder, png_name)
        image = Image.open(image_path).convert('L')
        print(type(image))
        print(image.size)
        image = image.crop((0, 0, 926, 926))
        print(image.size)
        image.save(output_path)
    """"
    # else:
    #     output_h5 = "/scratch/trueba/yeast/mitochondria/2Dmitochondria.h5"
    #     with h5py.File(output_h5, 'a') as f:
    #         image_path = join(folder_path, png_name)
    #         img = Image.open(image_path).convert('L')
    #         img = np.array(img)[:926, :926]
    #         f[path_vol] = img
    """
