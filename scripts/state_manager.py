import json 
import os

# Try to import redis, fallback to file-based storage if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

STATE_KEY = "state"


def _get_redis_client():
    """Get a Redis client connection with configuration from environment variables."""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD", None)
        db = int(os.getenv("REDIS_DB", 0))
        
        return redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
    except Exception:
        return None


def _get_file_state_path() -> str:
    """Get the file path for file-based state storage."""
    state_dir = os.path.join(os.path.dirname(__file__), "state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, "training_state.json")


def get_state(task_id: str | None = None) -> dict:
    """Get the state from Redis or file. If task_id is provided, use task-specific state."""
    if task_id:
        state_key = f"{STATE_KEY}_{task_id}"
    else:
        state_key = STATE_KEY
    
    # Try Redis first
    client = _get_redis_client()
    if client:
        try:
            value = client.get(state_key)
            if value is not None:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {}
        except Exception:
            pass  # Fall back to file storage
    
    # Fall back to file-based storage
    state_path = _get_file_state_path()
    if task_id:
        # For task-specific state, use a separate file
        state_path = os.path.join(os.path.dirname(state_path), f"training_state_{task_id}.json")
    
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    return {}


def set_state(state: dict, task_id: str | None = None) -> None:
    """Set the state in Redis or file. If task_id is provided, use task-specific state."""
    if task_id:
        state_key = f"{STATE_KEY}_{task_id}"
    else:
        state_key = STATE_KEY
    
    json_value = json.dumps(state, indent=2)
    
    # Try Redis first
    client = _get_redis_client()
    if client:
        try:
            client.set(state_key, json_value)
            return
        except Exception:
            pass  # Fall back to file storage
    
    # Fall back to file-based storage
    state_path = _get_file_state_path()
    if task_id:
        # For task-specific state, use a separate file
        state_path = os.path.join(os.path.dirname(state_path), f"training_state_{task_id}.json")
    
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            f.write(json_value)
    except IOError as e:
        print(f"Warning: Failed to save state to file: {e}", flush=True)


def clear_state(task_id: str | None = None) -> None:
    """Clear the state for a task or global state."""
    if task_id:
        state_key = f"{STATE_KEY}_{task_id}"
    else:
        state_key = STATE_KEY
    
    # Try Redis first
    client = _get_redis_client()
    if client:
        try:
            client.delete(state_key)
        except Exception:
            pass
    
    # Clear file-based storage
    state_path = _get_file_state_path()
    if task_id:
        state_path = os.path.join(os.path.dirname(state_path), f"training_state_{task_id}.json")
    
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
        except Exception:
            pass


def test():
    """Test function to verify state manager works."""
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))
    

if __name__ == "__main__":
    test()
