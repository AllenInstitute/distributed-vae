from time import sleep

def main():
  i = 0 
  while True:
      print(f"[{i}] running...")
      sleep(10)
      i += 1

if __name__ == "__main__":
  main()