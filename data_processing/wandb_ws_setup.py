import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # We use the Reports API for adding panels



def minMaxPannel(
        metric_name,
        y_title = "",
        title = None
    ):
    if title is None:
        title = metric_name.split("/")[-1] + " Distribution"
    return wr.LinePlot(
        title=title,
        x="Step",
        title_x="Steps",
        y=[
            metric_name + " (mean)",
            metric_name + " (max)",
            metric_name + " (min)"
        ],
        title_y= y_title,
        plot_type="line"

    )

if __name__ == "__main__":
    api = wandb.Api()
    wandb.login(
        key='a593b534585893ad93cf62243228a866c9053247',
        force=True
    )

    workspace = ws.Workspace(
        entity="hur",
        project="TB2_Early_Tests",
        name="IsaacLab RL Analysis Workspace"
    )
    workspace.settings.remove_legends_from_panels = True
    workspace.settings.max_runs = 1
    #workspace = ws.Workspace.from_url("https://wandb.ai/hur/Tester?nw=nwuserrobonuke") #?nw=mcemps4n97n")
    """print("Init Sections:")
    for i in workspace.sections:
        print("\t", i.name)
    workspace.sections.clear()
    print("After Clear")
    for i in workspace.sections:
        print("\t", i.name)"""
    

    sections = {
        "Episode Length-Final":[
            ("Evaluation Episode Length", "Eval Episode / Total Timesteps", "Steps"),
            ("Training Episode Length", "Training Episode / Total Timesteps", "Steps")
        ],
        "Termination Conditions-Final":[
            "success", "time_out", "peg_broke"
        ],
        "Training Reward-Final":[
            ("Avg Reward", "Training Reward / Instantaneous reward", "Reward"),
            ("Return", "Training Reward / Return", "Reward"),
            ("Avg Distance to Goal Reward", "Training Reward / Step dist_to_goal", "Reward"),
            ("Avg Success Reward","Training Reward / Step success", "Reward"),
            ("Avg Failure Reward", "Training Reward / Step failure", "Reward"),
            ("Avg Alignment Reward", "Training Reward / Step alignment", "Reward")
        ],
        "Evaluation Reward-Final":[
            ("Avg Reward", "Eval Reward / Instantaneous reward", "Reward"),
            ("Return", "Eval Reward / Return", "Reward"),
            ("Avg Distance to Goal Reward", "Eval Reward / Step dist_to_goal", "Reward"),
            ("Avg Success Reward","Eval Reward / Step success", "Reward"),
            ("Avg Failure Reward", "Eval Reward / Step failure", "Reward"),
            ("Avg Alignment Reward", "Eval Reward / Step alignment", "Reward")
        ],
        "Training Smoothness-Final":[
            #("Peg Force", "Training Smoothness / Force", "Force (N)"),
            ("Sum of Squared Velocity", "Training Smoothness / Step Squared Joint Velocity", "SSV"),# (m/s)^2"),
            ("Jerk", "Training Smoothness / Step Jerk", "Jerk "),#(m/s^2)^2")
        ],
        "Evaluation Smoothness-Final":[
            #("Peg Force", "Eval Smoothness / Force", "Force (N)"),
            ("Sum of Squared Velocity", "Eval Smoothness / Step Squared Joint Velocity", "SSV"),# (m/s)^2"),
            ("Jerk", "Eval Smoothness / Step Jerk", "Jerk"),# (m/s^2)^2")
        ],
        "Media-Final":[],
        "Loss + Std Dev-Final":[
            ("Action Sampling Std Dev", "Policy / Standard deviation", "Holder"),
            ("Entropy Loss", "Loss / Entropy loss", "Loss"),
            ("Policy Loss", "Loss / Policy loss", "Loss"),
            ("Value Loss", "Loss / Value loss", "Loss"),
        ]
    }


    for sect_title, sect_data in sections.items():
        new_sect = ws.Section(
            name=sect_title,
            is_open=True
        )
        print(sect_title)
        if sect_title == "Termination Conditions-Final" or sect_title == "Media-Final":
            print(f"\tSkipping {sect_title}")
            pass # do these by hand cuz they quick
        else:
            #print("\t", sect_data)
            for pan_title, metric_key, y_label in sect_data:
                print("\t", pan_title, metric_key, y_label)
                new_sect.panels.append(
                    minMaxPannel(
                        metric_key,
                        y_label,
                        pan_title
                    )
                )
        workspace.sections.append(new_sect)

    workspace.save_as_new_view()


