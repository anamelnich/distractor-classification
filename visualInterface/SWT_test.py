# receiver_tid.py
from cnbiloop import BCI_tid
import select
import sys

# Initialize your existing interface (non-blocking sockets, serializers, etc.)
bci = BCI_tid.BciInterface()

def receiveTiD():
    global bci
    data = None
    try:
        data = bci.iDsock_bus.recv(512).decode("utf-8")
        bci.idStreamer_bus.Append(data)
    except BlockingIOError as e:
        if e.errno != 11:
            print("BlockingIOError in receiveTiD:", e)
    except Exception as e:
        print("Error in receiveTiD:", e)
    if data:
        if bci.idStreamer_bus.Has("<tobiid", "/>"):
            msg = bci.idStreamer_bus.Extract("<tobiid", "/>")
            bci.id_serializer_bus.Deserialize(msg)
            bci.idStreamer_bus.Clear()
            return int(round(float(bci.id_msg_bus.GetEvent())))
        elif bci.idStreamer_bus.Has("<tcstatus", "/>"):
            # drain unwanted messages
            count = bci.idStreamer_bus.Count("<tcstatus")
            for _ in range(1, count-1):
                bci.idStreamer_bus.Extract("<tcstatus", "/>")
    return None


def listen_and_print(loop_hz: float = 50.0):
    """
    Polls the socket and prints every TiD event value as it's received.
    """
    import time
    interval = 1.0 / max(1.0, loop_hz)

    print("Listening for TiD events...")
    try:
        while True:
            for value in receiveTiD():
                # Print exactly what we parsed; this will include 0 as well
                print(f"TiD event: {value}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    # Run as: python receiver_tid.py
    listen_and_print()
