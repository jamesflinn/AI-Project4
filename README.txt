Anthony Dario (ard74) James Flinn (jrf116)
EECS 391 Programming Assignment 4

This is an agent that uses Q-learning to play SEPIA. Our agent assigns actions to the units whenever an event occurs.
Events are when a friendly unit is attacked or when a unit dies. This tends to be between 25-35 times per episode. The
action that the footman is given is the action that maximizes the Q-value of the footman.

Our feature vector includes 5 features detailed below:

NUM_ATTACKING_FOOTMEN_FEATURE:
    This is the number of footmen attacking the the potential target. This is useful for coordinating multiple footmen
    to focus on a single footmen. This will hopefully allow our agent to focus units down lowering the amount of damage
    the enemy can do.

BEING_ATTACKED_FEATURE:
    This feature determines if the agent is being attacked by the potential target. Ideally this will promote agents to
    defend themselves.

CLOSEST_ENEMY_FEATURE:
    This feature is if the enemy is the closest enemy to the footman. We want to attack the closest enemy so we don't
    waste time walking around being attacked by enemies.

HEALTH_FEATURE:
    This is the difference in health between the footman and the victim. We want to attack units with less health than
    us so that we can win the 1v1 fight.

WEAKEST_ENEMY_FEATURE:
    This feature is if the victim is the weakest enemy currently. We want to gang up on the weakest enemy so that we can
    lower the number of enemies as quickly as possible.

VICTIM_HEALTH_FEATURE:
    This is the health of the friendly footman that is being attacked by the potential target. We want to save our units
    from dying.

Our agent tends to win 60% of the time in the 5v5 scenario with 1000 runs and 95% of the time in the 10v10 scenario with
1000 runs. As the number of runs goes up the agent tends to win more.
