import numpy as np
import onemath_v2c as v2c
import time
import sys

def run():
    run_row_major = True

    velo_size = 100000
    if len(sys.argv)>1:
        velo_size = int(sys.argv[1])

    if len(sys.argv)>2 and sys.argv[2]=='cm':
        run_row_major = False

    rng = np.random.default_rng()
    velo=np.array(rng.uniform(size=(4,velo_size)),np.float32)
    trf =np.array(rng.uniform(size=(4,4)),np.float32)
    rect=np.array(rng.uniform(size=(4,4)),np.float32)
    p2  =np.array(rng.uniform(size=(3,4)),np.float32)

    num_iters = np.clip(100000000//velo_size, 20, 5000)
    print(f'velo_size = {velo_size} num_iters = {num_iters}')

    out_np = p2.dot(rect.dot(trf.dot(velo)))
    start = time.perf_counter_ns()
    for iter in range(num_iters):
        out_np = p2.dot(rect.dot(trf.dot(velo)))
    end = time.perf_counter_ns()
    print(f'numpy time: {(end-start)/1e6/num_iters:.3f} ms')

    if run_row_major:
        out_v2c_rm = v2c.velo2cam_rm(velo,trf,rect,p2)
        start = time.perf_counter_ns()
        for iter in range(num_iters):
            out_v2c_rm = v2c.velo2cam_rm(velo,trf,rect,p2)
        end = time.perf_counter_ns()
        print(f'row-major time: {(end-start)/1e6/num_iters:.3f} ms')

    out_v2c_cm = v2c.velo2cam_cm(velo,trf,rect,p2)
    start = time.perf_counter_ns()
    for iter in range(num_iters):
        out_v2c_cm = v2c.velo2cam_cm(velo,trf,rect,p2)
    end = time.perf_counter_ns()
    print(f'col-major time: {(end-start)/1e6/num_iters:.3f} ms')

    def test_match(input,input_name,ref,ref_name):
        if input.shape!=ref.shape:
            print(f'== FAIL == {input_name} and {ref_name} have different dims:'
                  f' {input.shape} != {ref.shape}')
            return False
        if (np.abs(input-ref) > 1e-4).any():
            print(f'== FAIL == {input_name} and {ref_name} have different values:'
                  f' max abs diff = {(np.abs(input-ref).max())}')
            return False
        print(f'== PASS == {input_name} and {ref_name} match')
        return True


    test_match(out_v2c_cm, 'out_col_major', out_np, 'out_ref')
    if run_row_major:
        test_match(out_v2c_rm, 'out_row_major', out_np, 'out_ref')
        test_match(out_v2c_rm, 'out_row_major', out_v2c_cm, 'out_col_major')


if __name__ == '__main__':
    run()

