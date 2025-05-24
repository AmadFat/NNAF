# if __name__ == "__main__":
#     from nnaf.data.in100 import in100_wds_dali_loader
#     from alive_progress import alive_bar
#     from matplotlib import pyplot as plt
#     from matplotlib import gridspec
#     import numpy as np
#     import os
    


#     # Check if display is available
#     # if 'DISPLAY' in os.environ:
#     # try:
#     plt.switch_backend('Qt5Agg')
#     interactive = True
#         # except ImportError:
#         #     import matplotlib
#         #     matplotlib.use('Agg')
#     #         interactive = False
#     # else:
#     #     import matplotlib
#     #     matplotlib.use('Agg')
#     #     interactive = False

#     batch_size = 12

#     pipe = in100_wds_dali_loader(
#         "/home/af/Data/ImageNet100",
#         part="train",
#         batch_size=batch_size,
#         num_workers=32,
#     )
    
#     def show_images(image_batch, labels):
#         columns = 4
#         rows = (batch_size + 1) // (columns)
#         fig = plt.figure(figsize=(32, (32 // columns) * rows))
#         gs = gridspec.GridSpec(rows, columns)
#         for j in range(min(rows * columns, batch_size)):
#             plt.subplot(gs[j])
#             plt.axis("off")
#             ascii = labels.at(j)
#             plt.title(
#                 "".join([chr(item) for item in ascii]), fontdict={"fontsize": 25}
#             )
#             img_chw = image_batch.at(j)
#             img_hwc = np.transpose(img_chw, (1, 2, 0)) / 255.0
#             plt.imshow(img_hwc)
        
#         if interactive:
#             plt.show()
#         else:
#             plt.savefig("imagenet100_sample.png", bbox_inches='tight', dpi=150)
#             plt.close()
#             print("Saved visualization to imagenet100_sample.png")

#     img, ann = pipe.run()
#     show_images(img.as_cpu(), ann.as_cpu())

if __name__ == "__main__":
    from nnaf.data.voc07 import build_voc07_wds
    from nnaf.data.in100 import build_in100_wds

    # build_voc07_wds("/home/af/Data/VOC07")
    build_in100_wds("/home/af/Data/ImageNet100")