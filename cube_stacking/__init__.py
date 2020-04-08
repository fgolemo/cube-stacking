from cube_stacking.sim import RandomPositions
from gym import register

for randomness in [name for name, _ in RandomPositions.__members__.items()]:
    for headlessness in ["Headless", "Graphical"]:
        register(
            id=f'Cubestacc-SinglePlayer-Heuristic-ArenaFour-{randomness}-{headlessness}-v0',
            entry_point='cube_stacking.envs:SinglePlayer',
            kwargs={
                "headless": (True if headlessness == "Headless" else False),
                "arena": 4,
                "randomness": randomness
            },
            max_episode_steps=15,
        )

        register(
            id=f'Cubestacc-SinglePlayer-Heuristic-ArenaOne-{randomness}-{headlessness}-v0',
            entry_point='cube_stacking.envs:SinglePlayer',
            kwargs={
                "headless": (True if headlessness == "Headless" else False),
                "arena": 1,
                "randomness": randomness
            },
            max_episode_steps=15,
        )

        register(
            id=f'Cubestacc-SinglePlayer-Heuristic-RelativeAct-{randomness}-{headlessness}-v0',
            entry_point='cube_stacking.envs:SinglePlayer',
            kwargs={
                "headless": (True if headlessness == "Headless" else False),
                "arena": 4,
                "randomness": randomness,
                "relative_action": True
            },
            max_episode_steps=15,
        )

        for drl in ["PPO", "TD3"]:
            register(
                id=f'Cubestacc-TwoPlayer-Full-{randomness}-{drl}-{headlessness}-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "randomness": randomness,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True
                },
                max_episode_steps=10,
            )
            register(
                id=f'Cubestacc-TwoPlayer-Full-{randomness}-{drl}-{headlessness}-Eval-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "randomness": randomness,
                    "eval": True,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True
                },
                max_episode_steps=10,
            )
            register(
                id=f'Cubestacc-TwoPlayer-Full-H4-{randomness}-{drl}-{headlessness}-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "max_height": 4,
                    "randomness": randomness,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True
                },
                max_episode_steps=4,
            )
            register(
                id=f'Cubestacc-TwoPlayer-Full-H4-Textured-{randomness}-{drl}-{headlessness}-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "max_height": 4,
                    "randomness": randomness,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True,
                    "textured": True
                },
                max_episode_steps=4,
            )
            register(
                id=f'Cubestacc-TwoPlayer-Full-H4-{randomness}-{drl}-{headlessness}-Eval-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "max_height": 4,
                    "randomness": randomness,
                    "eval": True,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True
                },
                max_episode_steps=4,
            )
            register(
                id=f'Cubestacc-TwoPlayer-Full-Weighted-{randomness}-{drl}-{headlessness}-v2',
                entry_point='cube_stacking.envs:TwoPlayerDirectoryFullWeightedArena',
                kwargs={
                    "headless": (True if headlessness == "Headless" else False),
                    "arena": 4,
                    "randomness": randomness,
                    "drl": drl.lower(),
                    "reward_scheme": 2,
                    "no_floor": True
                },
                max_episode_steps=10,
            )

