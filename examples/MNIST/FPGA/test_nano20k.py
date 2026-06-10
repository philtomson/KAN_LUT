import json
import struct
import serial
import sys
import time

def load_luts(json_path, d_out, d_in):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    luts_dict = data['luts']
    luts_mat = {}
    for q in range(d_out):
        for p in range(d_in):
            key = f"lut_{q+1}_{p+1}"
            luts_mat[(q, p)] = luts_dict[key]
    
    # Pack into bytes (little-endian 16-bit signed or unsigned? The sim uses uint8_t for output but LUT values are 16-bit signed)
    # The verilog memory is 16-bit wide
    bin_data = bytearray()
    for q in range(d_out):
        for p in range(d_in):
            vals = luts_mat[(q, p)]
            for v in range(256):
                val = vals[v]
                if val < 0:
                    val = (1 << 16) + val
                bin_data.extend(struct.pack('<H', val))
    return bin_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_nano20k.py <serial_port>")
        sys.exit(1)
        
    port = sys.argv[1]
    baud = 2000000
    
    print("Packing weights...")
    l1_bin = load_luts('../examples/MNIST/mnist_luts_layer1.json', 64, 196)
    l2_bin = load_luts('../examples/MNIST/mnist_luts_layer2.json', 10, 64)
    
    weights = l1_bin + l2_bin
    expected_size = 6750208
    print(f"Total weight bytes: {len(weights)} (Expected: {expected_size})")
    assert len(weights) == expected_size
    
    print(f"Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=1)
    
    print("Waiting 1s for board boot...")
    time.sleep(1.0)
    
    print("Streaming weights to FPGA...")
    # Stream in chunks
    chunk_size = 65536
    start_time = time.time()
    for i in range(0, len(weights), chunk_size):
        chunk = weights[i:i+chunk_size]
        ser.write(chunk)
        print(f"  Sent {i + len(chunk)} / {len(weights)} bytes...", end='\r')
    
    ser.flush()
    elapsed = time.time() - start_time
    print(f"\nWeights loaded in {elapsed:.2f} seconds ({len(weights)/elapsed/1024:.2f} KB/s)!")
    
    # Now run inference on tb_data.txt
    print("Loading test data...")
    correct = 0
    total = 0
    with open('../examples/MNIST/FPGA/tb_data.txt', 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split(',')
            label = int(parts[0])
            img_bytes = bytearray([int(x) for x in parts[1:]])
            assert len(img_bytes) == 196
            
            # Send image
            ser.write(img_bytes)
            
            # Read 1 byte result
            res = ser.read(1)
            if len(res) == 1:
                pred = res[0]
                total += 1
                if pred == label:
                    correct += 1
                print(f"Label: {label}, Pred: {pred} | Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
            else:
                print("Timeout waiting for FPGA response!")
                
if __name__ == '__main__':
    main()
