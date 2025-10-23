from register_user import capture_and_recognize

def main():
    print("[INFO] Face Registration & Recognition System")
    print("Type 'exit' to quit at any time.\n")

    while True:
        username = input("Enter new username to register: ").strip()
        if username.lower() == 'exit':
            print("[INFO] Exiting...")
            break
        elif username == "":
            print("[WARNING] Username cannot be empty.")
            continue

        print(f"[INFO] Starting registration for user: {username}")
        try:
            capture_and_recognize(username)
            print(f"[INFO] Completed registration for {username}\n")
        except Exception as e:
            print(f"[ERROR] Failed to register user {username}: {e}\n")

if __name__ == "__main__":
    main()
