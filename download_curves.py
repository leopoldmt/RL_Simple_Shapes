import wandb
import pandas as pd


if __name__ == '__main__':

    runs = {'attr': ['ij7jwlkp', '7mcxdcn8', '6wk85zm6', '3s1qq72u', 'zhulatov'],
            'v': [],
            'GWattr': [],
            'GWv': [],
            'GWsupattr': [],
            'GWsupv': [],
            'CLIPattr': [],
            'CLIPv': []
    }

    # Specify the W&B run you want to fetch data from
    run_id = "ij7jwlkp"
    entity = "leopold_m"
    project = "RL_factory"

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Access the run's history
    history = run.history()

    # Convert the history to a Pandas DataFrame
    df = pd.DataFrame(history)

    # Specify the path to save the CSV file
    csv_filename = "wandb_data.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    # Close the W&B run
    wandb.finish()