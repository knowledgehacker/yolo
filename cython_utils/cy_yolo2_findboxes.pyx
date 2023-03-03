import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from box import BoundBox
from nms cimport NMS

#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y

#MAX
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float max_c(float a, float b):
    if(a>b):
        return a
    return b

"""
#SOFTMAX!
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _softmax_c(float* x, int classes):
    cdef:
        float sum = 0
        np.intp_t k
        float arr_max = 0
    for k in range(classes):
        arr_max = max(arr_max,x[k])
    
    for k in range(classes):
        x[k] = exp(x[k]-arr_max)
        sum += x[k]

    for k in range(classes):
        x[k] = x[k]/sum
"""
        
        

#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_constructor(meta,np.ndarray[float,ndim=3] net_out_in, float threshold):
    cdef:
        np.intp_t H, W, BxCCC, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W, BxCCC = meta['out_size']
    C = meta['classes']
    B = meta['num']
    
    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, BxCCC/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, :C]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, C:]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 0] = expit_c(Bbox_pred[row, col, box_loop, 0])
                Bbox_pred[row, col, box_loop, 1] = (col + expit_c(Bbox_pred[row, col, box_loop, 1])) / W # center_x / image_w
                Bbox_pred[row, col, box_loop, 2] = (row + expit_c(Bbox_pred[row, col, box_loop, 2])) / H # center_y / image_h
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 0] / W # w / image_w
                Bbox_pred[row, col, box_loop, 4] = exp(Bbox_pred[row, col, box_loop, 4]) * anchors[2 * box_loop + 1] / H # h / image_h
                for class_loop in range(C):
                    tempc = expit_c(Classes[row, col, box_loop, class_loop]) * Bbox_pred[row, col, box_loop, 0]
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc
    
    #NMS                    
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
