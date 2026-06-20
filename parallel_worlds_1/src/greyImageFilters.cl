        /* start the kernel with one work-item per pixel
        ** first work-dimension (0) is image width (x)
        */
        __kernel void grey(__global unsigned char *inImg,
                            __global unsigned char *outImg,
                            __private unsigned int w,__private unsigned int h) {
            __private unsigned int x;__private unsigned int y;
            x = get_global_id(0);
            y = get_global_id(1);
            if(y<h) {
                if(x<w) {
                    // greyscale conversion (c.f. http://en.wikipedia.org/wiki/Grayscale)
                    // Y = 0.2126R + 0.7152G + 0.0722B
                    outImg[x+w*y] = 0.0722 * inImg[3*(x+w*y)] /* blue */
                    + 0.7152 * inImg[3*(x+w*y)+1]             /* green */
                    + 0.2126 * inImg[3*(x+w*y)+2];            /* red */
                }
            }
        }
