from python_client import Trigger
import time

try:
        HWTrigger = Trigger('USB2LPT')
        print("Trigger initialized with USB2LPT.")
except Exception as e1:
        print("USB2LPT trigger failed:", e1)
        try:
            HWTrigger = Trigger('ARDUINO')
            print("Trigger initialized with ARDUINO.")
        except Exception as e2:
            print("ARDUINO trigger also failed:", e2)
            print("ERROR: No valid trigger could be initialized.")

HWTrigger.init(50)

codes = [4,8,32,44,64] # fixtion, trial start (RD,LD,ND), feedback = 5 triggers
for i in range(10): 
	for code in codes:
		HWTrigger.signal(code)
		time.sleep(0.35)

