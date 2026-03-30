import numpy as np
import unittest

from deep_learning import Conv1D, Conv2D


class TestConv(unittest.TestCase):

    def test_conv1d_channels1(self):
        x = np.random.randn(1, 12).astype(np.float32)
        conv1 = Conv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        output = conv1(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    
    def test_conv1d_channels2(self):
        x = np.random.randn(2,12).astype(np.float32)
        conv1 = Conv1D(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0)
        output = conv1(x)
        assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"

    def test_conv1d_channels3(self):
        x = np.random.randn(3,12).astype(np.float32)
        conv1 = Conv1D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        output = conv1(x)
        assert output.shape == (6, 10), f"Expected output shape (6, 10), got {output.shape}"
        
    def test_conv1d_simple(self):
        x = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        conv1 = Conv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        weight = np.array([[[1, 0, -1]]], dtype=np.float32)
        conv1.set_weight(weight)
        output1 = conv1(x)
        assert output1.shape == (1, 3), f"Expected output shape (1, 3), got {output1.shape}"
        assert np.allclose(output1, np.array([[-2, -2, -2]])), f"Expected output [[-2, -2, -2]], got {output1}"

    def test_conv1d_complex(self):
        x = np.array([[1,3,4,2,1,7,8],[3,1,2,7,5,4,2],[1,2,3,4,5,6,7]], dtype=np.float32)
        conv1 = Conv1D(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
        weight = np.array([[[ 0.33333333,  0.33333333,  0.33333333],
                            [ 0.33333333,  0.33333333,  0.33333333],
                            [ 0.33333333,  0.33333333,  0.33333333]],
   
                           [[ 0.        ,  1.        ,  0.        ],
                            [ 0.        ,  1.        ,  0.        ],
                            [ 0.        ,  1.        ,  0.        ]],
                    
                           [[-1.        ,  0.        ,  1.        ],
                            [-1.        ,  0.        ,  1.        ],
                            [-1.        ,  0.        ,  1.        ]],
                    
                           [[ 0.        ,  0.5       ,  1.        ],
                            [ 0.        ,  1.        ,  2.        ],
                            [ 0.        ,  1.5       ,  3.        ]],
                    
                           [[ 1.        ,  0.5       ,  0.        ],
                            [ 2.        ,  1.        ,  0.        ],
                            [ 3.        ,  1.5       ,  0.        ]]], dtype=np.float32)
        conv1.set_weight(weight)
        output1 = conv1(x)
        assert output1.shape == (5, 5), f"Expected output shape (5, 5), got {output1.shape}"
        res = np.array([[ 6.6667,  9.3333, 11.0000, 13.6667, 15.0000],
                        [ 6.0000,  9.0000, 13.0000, 11.0000, 17.0000],
                        [ 4.0000,  7.0000,  2.0000,  4.0000,  6.0000],
                        [22.5000, 36.5000, 40.0000, 46.0000, 49.5000],
                        [15.5000, 19.5000, 31.0000, 41.0000, 42.5000]], dtype=np.float32)
        assert np.allclose(output1, res), f"Expected output {res}, got {output1}"

    def test_conv2d_channels1(self):
        conv2d = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        x = np.ones((7,5,1))
        output = conv2d(x)
        assert output.shape == (1, 5, 3), f"Expected output shape (1, 5, 3), got {output.shape}"

    def test_conv2d_channels2(self):
        conv2d = Conv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        x = np.ones((7,5,1))
        output = conv2d(x)
        assert output.shape == (3, 5, 3), f"Expected output shape (3, 5, 3), got {output.shape}"

    def test_conv2d_simple(self):
        conv2d = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        x = np.ones((7,5,1))
        weight = np.array([[[[1,0,0], 
                            [0,1,0], 
                            [0,0,1]]]])
        conv2d.set_weight(weight)
        output = conv2d(x)
        true_output = np.array([[[2, 2, 2, 2, 1,],
                                 [2, 3, 3, 3, 2,],
                                 [2, 3, 3, 3, 2,],
                                 [2, 3, 3, 3, 2,],
                                 [2, 3, 3, 3, 2,],
                                 [2, 3, 3, 3, 2,],
                                 [1, 2, 2, 2, 2,]]])
        assert np.allclose(output, true_output), f"Expected output {true_output}, got {output}"

    def test_conv2d_complex(self):
        x = np.array([
    [[ 1,  2,  3], [ 4,  5,  6], [ 7,  8,  9], [10, 11, 12], [13, 14, 15]],
    [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
    [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42], [43, 44, 45]],
    [[46, 47, 48], [49, 50, 51], [52, 53, 54], [55, 56, 57], [58, 59, 60]],
    [[61, 62, 63], [64, 65, 66], [67, 68, 69], [70, 71, 72], [73, 74, 75]]
    ], dtype=np.float32)
        print(x.shape)
        kernel = np.array([[
    [[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]],
    [[ 2,  0, -2], [ 2,  0, -2], [ 2,  0, -2]],
    [[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]]
    ]], dtype=np.float32)
        conv2d = Conv2D(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)
        conv2d.set_weight(kernel)
        output = conv2d(x)
        true_output = out = np.array([
            [-72., -72., -72.],
            [-72., -72., -72.],
            [-72., -72., -72.]
            ], dtype=np.float32)
        assert np.allclose(output, true_output), f"Expected {true_output}, got {output}"

if __name__ == "__main__":
    unittest.main()