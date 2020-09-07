import argparse, os, sys
import caffe
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototxt", type=str, default="./centernet_dcnv2.prototxt")
    parser.add_argument("--caffemodel", type=str, default="./centernet_dcnv2.caffemodel")
    args = parser.parse_args(sys.argv[1:])
    data = np.load('centernet_data.npy')
    import pdb; pdb.set_trace()
    #data = np.random.randn(1,3,226,226)
    #np.save('input_alex',data)
    caffe.set_mode_gpu()
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.blobs["blob1"].reshape(1, 3, 512, 512)
    net.blobs["blob1"].data[...] = data
    output = net.forward()
    import pdb; pdb.set_trace()
    np.savetxt('out_caffe', output['deconv_blob1'][0,0,:,:])
    print("Successed!")
