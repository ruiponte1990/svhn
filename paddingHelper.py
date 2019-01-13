import numpy as np
# a function to help us stay clean
def getPaddings(pad_along_height,pad_along_width):
    # if even.. easy..
    if pad_along_height%2 == 0:
        pad_top = pad_along_height / 2
        pad_bottom = pad_top
    # if odd
    else:
        pad_top = np.floor( pad_along_height / 2 )
        pad_bottom = np.floor( pad_along_height / 2 ) +1
    # check if width padding is odd or even
    # if even.. easy..
    if pad_along_width%2 == 0:
        pad_left = pad_along_width / 2
        pad_right= pad_left
    # if odd
    else:
        pad_left = np.floor( pad_along_width / 2 )
        pad_right = np.floor( pad_along_width / 2 ) +1
        #
    return pad_top,pad_bottom,pad_left,pad_right

# strides [image index, y, x, depth]
# padding 'SAME' or 'VALID'
# bottom and right sides always get the one additional padded pixel (if padding is odd)
def getOutputDim (inputWidth,inputHeight,filterWidth,filterHeight,strides,padding):
    if padding == 'SAME':
        out_height = np.ceil(float(inputHeight) / float(strides[1]))
        out_width  = np.ceil(float(inputWidth) / float(strides[2]))
        #
        pad_along_height = ((out_height - 1) * strides[1] + filterHeight - inputHeight)
        pad_along_width = ((out_width - 1) * strides[2] + filterWidth - inputWidth)
        #
        # now get padding
        pad_top,pad_bottom,pad_left,pad_right = getPaddings(pad_along_height,pad_along_width)
        #
        print('output height', out_height)
        print('output width' , out_width)
        print('total pad along height' , pad_along_height)
        print('total pad along width' , pad_along_width)
        print('pad at top' , pad_top)
        print('pad at bottom' ,pad_bottom)
        print('pad at left' , pad_left)
        print('pad at right' ,pad_right)

    elif padding == 'VALID':
        out_height = np.ceil(float(inputHeight - filterHeight + 1) / float(strides[1]))
        out_width  = np.ceil(float(inputWidth - filterWidth + 1) / float(strides[2]))
        #
        print('output height', out_height)
        print('output width' , out_width)
        print('no padding')


# use like so
getOutputDim (128,128,8,8,[1,1,1,1],'SAME')