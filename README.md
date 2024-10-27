# Image_processing_binary_mask
Given a set of jpg photo files on disk, write code using OpenCV that would, in parallel, read each in,  create a binary mask image where max means all 3 channels are above 200 (considering 8 bit as max), and write out  the mask images as any lossless files. Also, sum the number of pixels where the mask is max in all images  and log that count.
