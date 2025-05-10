import os
from pathlib import Path

def process_keypoints(line):
    parts = list(map(float, line.strip().split()))
    if len(parts) != 1 + 4 + 3 * 8:  # class_id + 8 keypoints (x,y,visibility)
        return None
    
    class_id = int(parts[0])
    keypoints = parts[5:]
    
    # Extract all original keypoints (x, y, visibility)
    kps = [(keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]) for i in range(8)]
    
    # Calculate new keypoints according to requirements:
    # 1st: average of 8th and 1st points (indices 7 and 0)
    kp1 = [(kps[7][0] + kps[0][0])/2, (kps[7][1] + kps[0][1])/2, int(min(kps[7][2], kps[0][2]))]
    
    # 2nd: average of 2nd and 3rd points (indices 1 and 2)
    kp2 = [(kps[1][0] + kps[2][0])/2, (kps[1][1] + kps[2][1])/2, int(min(kps[1][2], kps[2][2]))]
    
    # 3rd: average of 4th and 5th points (indices 3 and 4)
    kp3 = [(kps[3][0] + kps[4][0])/2, (kps[3][1] + kps[4][1])/2, int(min(kps[3][2], kps[4][2]))]
    
    # 4th: average of 6th and 7th points (indices 5 and 6)
    kp4 = [(kps[5][0] + kps[6][0])/2, (kps[5][1] + kps[6][1])/2, int(min(kps[5][2], kps[6][2]))]
    
    # Combine new keypoints (class_id followed by x,y,visibility for each point)
    new_keypoints = [class_id] + parts[1:5] + [coord for kp in [kp1, kp2, kp3, kp4] for coord in kp]
    
    return ' '.join(map(str, new_keypoints))

def process_labels():
    input_dir = "/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_blender_v0.3/label_processed8"
    output_dir = "/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_blender_v0.3/true_labels"
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each .txt file in input directory
    for label_file in input_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        processed_lines = []
        for line in lines:
            processed_line = process_keypoints(line)
            if processed_line is not None:
                processed_lines.append(processed_line)
        
        # print(processed_lines)
        # Save to output directory with same filename
        output_file = output_path / label_file.name
        with open(output_file, 'w') as f:
            f.write('\n'.join(processed_lines))
    
    print(f"Processing complete. New labels saved to {output_dir}")

if __name__ == '__main__':
    process_labels()