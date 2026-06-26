import random
import math
import os
import json

def generate_guaranteed_tsptw_test_cases():
    output_dir = "tsptw_testcases_with_answers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary dùng để gom toàn bộ đáp án
    answers_data = {}
        
    # Tập hợp các giá trị N (Bình thường: 10 -> 950, Lớn: 1500 -> 5000)
    N_normal = [10] + [i * 50 for i in range(1, 21)]
    N_large = [1500 + i * 500 for i in range(3)]
    all_N = N_normal + N_large
    
    for idx, N in enumerate(all_N):
        grid_size = 1000 if N <= 1000 else 10000
        filename = os.path.join(output_dir, f"test_{idx+1:02d}_N_{N}.txt")
        
        # 1. Sinh tọa độ và ma trận khoảng cách t
        coords = [(random.randint(0, grid_size), random.randint(0, grid_size)) for _ in range(N + 1)]
        t = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    t[i][j] = int(round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])))
        
        # 2. Xây dựng lộ trình bí mật hợp lệ
        hidden_path = list(range(1, N + 1))
        random.shuffle(hidden_path)
        
        e_list = [0] * (N + 1)
        l_list = [0] * (N + 1)
        d_list = [0] * (N + 1)
        
        current_time = 0
        current_node = 0
        total_travel_cost = 0
        
        for next_node in hidden_path:
            arrival_time = current_time + t[current_node][next_node]
            
            wait_time = random.randint(0, 50)
            e_i = arrival_time + wait_time
            d_i = random.randint(5, 30)
            finish_time = e_i + d_i
            
            slack = random.randint(50, 300)
            l_i = finish_time + slack
            
            e_list[next_node] = int(e_i)
            l_list[next_node] = int(l_i)
            d_list[next_node] = int(d_i)
            
            current_time = finish_time
            total_travel_cost += t[current_node][next_node]
            current_node = next_node
            
        # 3. Ghi dữ liệu của bài toán vào file .txt
        with open(filename, 'w') as f:
            f.write(f"{int(N)}\n")
            for i in range(1, N + 1):
                f.write(f"{e_list[i]} {l_list[i]} {d_list[i]}\n")
            for row in t:
                f.write(" ".join(map(str, row)) + "\n")
                
        # 4. Ghi nhận cấu trúc lộ trình vào Dictionary
        # Key của JSON bắt buộc phải là dạng String (ví dụ: "10", "50")
        answers_data[str(N)] = {
            "cost": total_travel_cost,
            "route": hidden_path # Python List tự động chuyển đổi thành JSON Array
        }
            
        print(f"Đã tạo Test Case: N = {N}")

    # 5. Ghi toàn bộ Dictionary ra file JSON ở cuối tiến trình
    json_filename = os.path.join(output_dir, "answers.json")
    with open(json_filename, 'w') as json_file:
        # Tham số indent=4 giúp file JSON được định dạng đẹp, thụt lề rõ ràng dễ đọc
        json.dump(answers_data, json_file, indent=4)
        
    print(f"\n✅ Hoàn tất! File đáp án JSON đã được lưu tại: {json_filename}")

if __name__ == "__main__":
    generate_guaranteed_tsptw_test_cases()