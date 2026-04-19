"""Probe what your volume dial / media keys emit.

Run this, then:
  1. Twist the volume dial CLOCKWISE a few detents slowly.
  2. Twist COUNTER-CLOCKWISE a few detents slowly.
  3. Press the dial button (if it clicks / presses down).
  4. Press ESC to stop.

We log scan codes and names. If nothing prints when you twist the dial, the
keyboard routes volume through a vendor driver that bypasses the Python hook
and we'll need a different capture strategy.
"""
import time
import keyboard

print("probing... twist dial CW, CCW, press button, then ESC to stop\n")

events = []
start = time.time()


def log(e):
    dt = time.time() - start
    events.append((dt, e.name, e.scan_code, e.event_type))
    print(f"{dt:6.2f}s  name={e.name!r:<18} scan_code={e.scan_code:<6} type={e.event_type}")


keyboard.hook(log)
keyboard.wait("esc")

print("\n--- summary ---")
from collections import Counter
names = Counter(e[1] for e in events)
for name, count in names.most_common():
    print(f"  {name!r}: {count} events")
