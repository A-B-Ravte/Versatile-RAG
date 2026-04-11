from llama_cpp import Llama


class LlammaLocal:
    _instance = None    

    @classmethod
    def get_instance(cls, model_path: str):
        if cls._instance == None :
            print(f"Loading model from {model_path}... Please wait.")
            # n_ctx: The 'memory' of the conversation (2048 is standard)
            # n_threads: Number of CPU cores to use. 4 is usually a sweet spot for laptops.
            cls._instance = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False # Set to True if you want to see the 'math' in terminal
            )
            print("Model loaded successfully!")
        return cls._instance