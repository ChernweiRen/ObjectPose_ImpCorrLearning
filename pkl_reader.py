import pickle

with open("/data/user/rencw/ICL-I2PReg/views/plane_checker__test_tube_rack_alarm_clock_coffee_cup/meta.pkl", "rb") as f:
    meta = pickle.load(f)

print(meta)