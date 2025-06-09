from collections import deque

# Queue for keypoint frame handling
class FrameQueue:
    def __init__(self, max_length=30):
        self.queue = deque(maxlen=max_length)

    def add_frame(self, frame_data):
        self.queue.append(frame_data)

    def get_sequence(self):
        return list(self.queue)

# Stack for label management
class LabelStack:
    def __init__(self):
        self.stack = []

    def add_label(self, label):
        self.stack.append(label)

    def get_last_label(self):
        return self.stack[-1] if self.stack else None

    def pop_label(self):
        return self.stack.pop() if self.stack else None

    def get_all_labels(self):
        return list(self.stack)


# Test run - looks like project logic
if __name__ == "__main__":
    # Using FrameQueue
    frame_handler = FrameQueue()
    for i in range(100):  # Imagine these are real extracted keypoints
        frame_handler.add_frame(f"keypoints_frame_{i}")
    input_sequence = frame_handler.get_sequence()

    print("🟢 Input sequence (last 30 frames):")
    print(input_sequence)

    # Using LabelStack
    label_handler = LabelStack()
    label_handler.add_label("hello")
    label_handler.add_label("thanks")
    label_handler.add_label("iloveyou")

    print("\n🟢 Labels used (stack format):")
    print(label_handler.get_all_labels())

    last = label_handler.pop_label()
    print(f"\n🔴 Removed last label: {last}")
    print(f"🟢 Current top label: {label_handler.get_last_label()}")
