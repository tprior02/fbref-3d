seasons_dictionary = {
    '2023-24': 24,
    '2022-23': 23,
    '2021-22': 22,
    '2020-21': 21,
    '2019-20': 20,
    '2018-19': 19,
    '2017-18': 18}

competitions_dictionary = {
    'premier league': 'ENG',
    'serie A': 'ITA',
    'bundesliga': 'GER',
    'la liga': 'SPA',
    'ligue 1': 'FRA',
    'champions league': 'CL',
    'europa league': 'EL'}

tables_dictionary = {
    'shooting': 'shooting',
    'passing': 'passing',
    'pass types': 'passing_types',
    'shot creation': 'gca',
    'possession': 'possession',
    'defense': 'defense',
    'miscellaneous': 'misc',
    'playing time': 'playingtime',
    'keepers': 'keepers',
    'keepers advanced': 'keepersadv'
}

categories_list = [
    'defense',
    'gca',
    'keepers',
    'keepersadv',
    'misc',
    'passing',
    'passing_types',
    'playerinfo',
    'playingtime',
    'possession',
    'shooting'
]

categories_dictionary = {
    'shooting': 'shooting',
    'passing': 'passing',
    'pass types': 'passing_types',
    'shot creation': 'gca',
    'possession': 'possession',
    'defense': 'defense',
    'miscellaneous': 'misc',
    'playing time': 'playingtime',
    'keepers': 'keepers',
    'keepers advanced': 'keepersadv'
}

positions_list = [
    'goalkeeper',
    'centre-back',
    'full-back',
    'defensive midfield',
    'central midfield',
    'attacking midfield',
    'winger',
    'forward'
]

exclude_from_per_90_set = {
    'matches_gk',
    'starts_gk',
    'minutes_gk',
    'nineties_gk',
    'matches',
    'minutes',
    'nineties',
    'starts',
    'minutes',
    'pk_save_perc',
    'launch_completion',
    'launch_perc',
    'goal_kick_launch_perc',
    'cross_stop_perc',
    'pass_completion',
    'pass_completion_short',
    'pass_completion_medium',
    'pass_completion_long',
    'dribbler_tackle_perc',
    'take_on_success',
    'aerial_success',
    'avg_gk_pass_len',
    'avg_goal_kick_len',
    'avg_def_outside_pen_dist',
    'goal_per_shot',
    'goal_per_sot',
    'npxg_per_shot',
    'avg_shot_dist',
    'prog_passes_per_pass',
    'non_pen_goal_per_non_pen_shot',
    'opp_pen_pass_per_pass'
}

shooting_dictionary = {
    'non penalty goals': 'non_penalty_goals',
    'goals': 'goals',
    'shots': 'shot',
    'shots on target': 'shot_on_target',
    'freekicks': 'fk',
    'penalties scored': 'pk',
    'penalties attempted': 'att_pk',
    'xg': 'xg',
    'npxg': 'npxg',
    'goals - xg': 'goals_minus_xg',
    'npgoals - npxg': 'non_penalty_goals_minus_npxg',
    'goals / shot': 'goal_per_shot',
    'goals / shot on target': 'goal_per_sot',
    'npxg / shot': 'npxg_per_shot',
    'average shot distance': 'avg_shot_dist',
    'non penalty shots': 'non_pen_shots',
    'non penalty goals / non penalty shots': 'non_pen_goal_per_non_pen_shot'
}
passing_dictionary = {
    'completed passes': 'cmp_pass',
    'attempted passes': 'att_pass',
    'pass completion (%)': 'pass_completion',
    'total pass distance': 'tot_pass_dist',
    'progressive pass distance': 'prog_pass_dist',
    'completed short passes': 'cmp_pass_short',
    'attempted short passes': 'att_pass_short',
    'short pass completion (%)': 'pass_completion_short',
    'completed medium passes': 'cmp_pass_medium',
    'attempted medium passes': 'att_pass_medium',
    'medium pass completion (%)': 'medium_completion_short',
    'completed long passes': 'cmp_pass_long',
    'attempted long passes': 'att_pass_long',
    'long pass completion (%)': 'long_completion_short',
    'assists': 'assist',
    'xag': 'xag',
    'xa': 'xa',
    'assists - xag': 'assists_minus_xag',
    'key passes': 'key_pass',
    'passes to final third': 'fin_3rd_pass',
    'passes to penalty area': 'opp_pen_pass',
    'accurate crosses': 'acc_cross',
    'progressive passes': 'prog_pass',
    'progressive passes / completed pass': 'prog_passes_per_pass',
    'key passes / completed pass': 'key_pass_per_pass',
    'opposing penalty box pass / completed pass': 'opp_pen_pass_per_pass'
}
passing_types_dictionary = {
    'live passes': 'live_pass',
    'dead passes': 'dead_pass',
    'free kick passes': 'fk_pass',
    'completed through balls': 'tb_pass',
    'switch pass': 'sw_pass',
    'crosses': 'cross_pass',
    'throw ins': 'throw_in',
    'corner kicks': 'ck',
    'inswinging corner kicks': 'ck_in',
    'outswinging corner kicks': 'ck_out',
    'straight corner kicks': 'ck_straight',
    'offside passes': 'offside_pass',
    'blocked passes': 'blocked_pass'
}
gca_dictionary = {
    'shot creation': 'sca',
    'shot creation (live pass)': 'sca_pass_live',
    'shot creation (dead pass)': 'sca_pass_dead',
    'shot creation (take-on)': 'sca_take_on',
    'shot creation (shot)': 'sca_shot',
    'shot creation (fouled)': 'sca_fouled',
    'shot creation (defending)': 'sca_def',
    'goal creation': 'gca',
    'goal creation (live pass)': 'gca_pass_live',
    'goal creation (dead pass)': 'gca_pass_dead',
    'goal creation (take-on)': 'gca_take_on',
    'goal creation (shot)': 'gca_shot',
    'goal creation (fouled)': 'gca_fouled',
    'goal creation (defending)': 'gca_def',
}
possession_dictionary = {
    'touches': 'touch',
    'def pen area touches': 'touch_def_pen',
    'def 3rd touches': 'touch_def',
    'mid 3rd touches': 'touch_mid',
    'att 3rd touches': 'touch_att',
    'att pen area touches': 'touch_att_pen',
    'live ball touches': 'touch_live',
    'attempted take-ons': 'att_take_on',
    'completed take-ons': 'cmp_take_on',
    'take-on success (%)': 'take_on_success',
    'unsuccessful take-ons': 'uns_take_on',
    'carries': 'carry',
    'carry distance': 'carry_dist',
    'progressive carry distance': 'carry_prog_dist',
    'progressive carries': 'carry_prog',
    'carries to att 3rd': 'carry_att_third',
    'carries to att pen area': 'carry_opp_pen',
    'mis-controls': 'miscontrol',
    'dispossessed': 'disposs',
    'passes received': 'received',
    'progressive passes received': 'prog_received'
}
defense_dictionary = {
    'attempted tackles': 'att_tackle',
    'completed tackles': 'cmp_tackle',
    'attempted tackles (def 3rd)': 'att_tackle_def',
    'attempted tackles (mid 3rd)': 'att_tackle_mid',
    'attempted tackles (att 3rd)': 'att_tackle_att',
    'attempted dribbler tackles': 'att_drib_tackle',
    'completed dribbler tackles': 'cmp_drib_tackle',
    'dribbler tackle success (%)': 'dribbler_tackle_perc',
    'unsuccessful dribbler tackles': 'uns_drib_tackle',
    'blocks': 'block',
    'blocked shots': 'block_shot',
    'blocked passes': 'block_pass',
    'interceptions': 'intercept',
    'tackles + interceptions': 'tackle_plus_intercept',
    'clearances': 'clearance',
    'errors leading to shot': 'error'
}
misc_dictionary = {
    'yellow cards': 'y_card',
    'red cards': 'r_card',
    'two yellow cards': 'two_y_card',
    'fouls': 'fouls',
    'fouled': 'fouled',
    'offsides': 'offside',
    'penalties won': 'pens_won',
    'penalties conceded': 'pens_con',
    'own goals': 'own_goal',
    'posession recovered': 'recov',
    'aerials won': 'aerials_won',
    'aerials lost': 'aerials_lost',
    'aerials attempted': 'aerials_attempted',
    'aerial success (%)': 'aerial_success'
}
playingtime_dictionary = {
    'matches': 'matches',
    'minutes': 'minutes',
    'nineties': 'nineties',
    'starts': 'starts',
    'matches completed': 'completed',
    'substitute appearances': 'sub',
    'unused substitute': 'sub_unused',
    'team goals while on pitch': 'onpitch_goals',
    'team goals against while on pitch': 'onpitch_goals_ag',
    'team goals +/- while on pitch': 'onpitch_goals_delta',
    'team xg while on pitch': 'onpitch_xg',
    'team xga while on pitch': 'onpitch_xga',
    'team xg +/- while on pitch': 'onpitch_xg_delta',
}
keepers_dictionary = {
    'matches as goalkeeper': 'matches_gk',
    'starts as goalkeeper': 'starts_gk',
    'minutes as goalkeeper': 'minutes_gk',
    'nineties as goalkeeper': 'nineties_gk',
    'goals against': 'goals_ag',
    'shots on target against': 'sot_ag',
    'saves': 'saves',
    'save percentage (%)': 'save_perc',
    'wins as goalkeeper': 'gk_won',
    'draws as goalkeeper': 'gk_drew',
    'losses as goalkeeper': 'gk_lost',
    'clean sheets': 'cs',
    'clean sheet percentage (%)': 'clean_sheet_perc',
    'penalties faced': 'pk_att_ag',
    'penalties scored against': 'pk_scored_ag',
    'penalties saved': 'pk_saved',
    'penalty save percentage (%)': 'pk_save_perc',
    'penalties missed against': 'pk_missed_ag',
}
keepersadv_dictionary = {
    'free kicks against': 'fk_ag',
    'corner kicks against': 'ck_ag',
    'own goals against': 'og_ag',
    'post shot xg faced': 'ps_xg',
    'post shot xg +/-': 'ps_xg_delta',
    'attempted launches': 'att_launch',
    'completed launches': 'cmp_launch',
    'launch accuracy (%)': 'launch_completion',
    'passes launched (%)': 'launch_perc',
    'attempted keeper passes': 'att_gk_pass',
    'attempted keeper passes (exc. dead)': 'att_pass_non_goal_kick',
    'attempted throws': 'att_gk_throw',
    'avg keeper pass length (exc. dead)': 'avg_gk_pass_len',
    'goal kicks': 'att_goal_kick',
    'goal kick launches (%)': 'goal_kick_launch_perc',
    'short goal kicks': 'att_launch_non_goal_kick',
    'avg goal kick length': 'avg_goal_kick_len',
    'crosses faced': 'att_cross_ag',
    'crosses stopped': 'stop_cross_ag',
    'crosses stopped (%)': 'cross_stop_perc',
    'defensive actions outside pen': 'def_outside_pen',
    'average distance of defensive actions': 'avg_def_outside_pen_dist',
    'goal kicks launched': 'goal_kicks_launched'
}

category_to_table = {
    'att_tackle': 'defense',
    'cmp_tackle': 'defense',
    'att_tackle_def': 'defense',
    'att_tackle_mid': 'defense',
    'att_tackle_att': 'defense',
    'att_drib_tackle': 'defense',
    'cmp_drib_tackle': 'defense',
    'dribbler_tackle_perc': 'defense',
    'uns_drib_tackle': 'defense',
    'block': 'defense',
    'block_shot': 'defense',
    'block_pass': 'defense',
    'intercept': 'defense',
    'tackle_plus_intercept': 'defense',
    'clearance': 'defense',
    'error': 'defense',
    'sca': 'gca',
    'sca_pass_live': 'gca',
    'sca_pass_dead': 'gca',
    'sca_take_on': 'gca',
    'sca_shot': 'gca',
    'sca_fouled': 'gca',
    'sca_def': 'gca',
    'gca': 'gca',
    'gca_pass_live': 'gca',
    'gca_pass_dead': 'gca',
    'gca_take_on': 'gca',
    'gca_shot': 'gca',
    'gca_fouled': 'gca',
    'gca_def': 'gca',
    'matches_gk': 'keepers',
    'starts_gk': 'keepers',
    'minutes_gk': 'keepers',
    'nineties_gk': 'keepers',
    'goals_ag': 'keepers',
    'sot_ag': 'keepers',
    'saves': 'keepers',
    'save_perc': 'keepers',
    'gk_won': 'keepers',
    'gk_drew': 'keepers',
    'gk_lost': 'keepers',
    'cs': 'keepers',
    'clean_sheet_perc': 'keepers',
    'pk_att_ag': 'keepers',
    'pk_scored_ag': 'keepers',
    'pk_saved': 'keepers',
    'pk_save_perc': 'keepers',
    'pk_missed_ag': 'keepers',
    'fk_ag': 'keepersadv',
    'ck_ag': 'keepersadv',
    'og_ag': 'keepersadv',
    'ps_xg': 'keepersadv',
    'ps_xg_delta': 'keepersadv',
    'att_launch': 'keepersadv',
    'cmp_launch': 'keepersadv',
    'launch_completion': 'keepersadv',
    'launch_perc': 'keepersadv',
    'att_gk_pass': 'keepersadv',
    'att_pass_non_goal_kick': 'keepersadv',
    'att_gk_throw': 'keepersadv',
    'avg_gk_pass_len': 'keepersadv',
    'att_goal_kick': 'keepersadv',
    'goal_kick_launch_perc': 'keepersadv',
    'att_launch_non_goal_kick': 'keepersadv',
    'avg_goal_kick_len': 'keepersadv',
    'att_cross_ag': 'keepersadv',
    'stop_cross_ag': 'keepersadv',
    'cross_stop_perc': 'keepersadv',
    'def_outside_pen': 'keepersadv',
    'avg_def_outside_pen_dist': 'keepersadv',
    'goal_kicks_launched': 'keepersadv',
    'y_card': 'misc',
    'r_card': 'misc',
    'two_y_card': 'misc',
    'fouls': 'misc',
    'fouled': 'misc',
    'offside': 'misc',
    'pens_won': 'misc',
    'pens_con': 'misc',
    'own_goal': 'misc',
    'recov': 'misc',
    'aerials_won': 'misc',
    'aerials_lost': 'misc',
    'aerials_attempted': 'misc',
    'aerial_success': 'misc',
    'cmp_pass': 'passing',
    'att_pass': 'passing',
    'pass_completion': 'passing',
    'tot_pass_dist': 'passing',
    'prog_pass_dist': 'passing',
    'cmp_pass_short': 'passing',
    'att_pass_short': 'passing',
    'pass_completion_short': 'passing',
    'cmp_pass_medium': 'passing',
    'att_pass_medium': 'passing',
    'medium_completion_short': 'passing',
    'cmp_pass_long': 'passing',
    'att_pass_long': 'passing',
    'long_completion_short': 'passing',
    'assist': 'passing',
    'xag': 'passing',
    'xa': 'passing',
    'assists_minus_xag': 'passing',
    'key_pass': 'passing',
    'fin_3rd_pass': 'passing',
    'opp_pen_pass': 'passing',
    'acc_cross': 'passing',
    'prog_pass': 'passing',
    'prog_passes_per_pass': 'passing',
    'live_pass': 'passing_types',
    'dead_pass': 'passing_types',
    'fk_pass': 'passing_types',
    'tb_pass': 'passing_types',
    'sw_pass': 'passing_types',
    'cross_pass': 'passing_types',
    'throw_in': 'passing_types',
    'ck': 'passing_types',
    'ck_in': 'passing_types',
    'ck_out': 'passing_types',
    'ck_straight': 'passing_types',
    'offside_pass': 'passing_types',
    'blocked_pass': 'passing_types',
    'matches': 'playingtime',
    'minutes': 'playingtime',
    'nineties': 'playingtime',
    'starts': 'playingtime',
    'completed': 'playingtime',
    'sub': 'playingtime',
    'sub_unused': 'playingtime',
    'onpitch_goals': 'playingtime',
    'onpitch_goals_ag': 'playingtime',
    'onpitch_goals_delta': 'playingtime',
    'onpitch_xg': 'playingtime',
    'onpitch_xga': 'playingtime',
    'onpitch_xg_delta': 'playingtime',
    'touch': 'possession',
    'touch_def_pen': 'possession',
    'touch_def': 'possession',
    'touch_mid': 'possession',
    'touch_att': 'possession',
    'touch_att_pen': 'possession',
    'touch_live': 'possession',
    'att_take_on': 'possession',
    'cmp_take_on': 'possession',
    'take_on_success': 'possession',
    'uns_take_on': 'possession',
    'carry': 'possession',
    'carry_dist': 'possession',
    'carry_prog_dist': 'possession',
    'carry_prog': 'possession',
    'carry_att_third': 'possession',
    'carry_opp_pen': 'possession',
    'miscontrol': 'possession',
    'disposs': 'possession',
    'received': 'possession',
    'prog_received': 'possession',
    'non_penalty_goals': 'shooting',
    'goals': 'shooting',
    'shot': 'shooting',
    'shot_on_target': 'shooting',
    'fk': 'shooting',
    'pk': 'shooting',
    'att_pk': 'shooting',
    'xg': 'shooting',
    'npxg': 'shooting',
    'goals_minus_xg': 'shooting',
    'non_penalty_goals_minus_npxg': 'shooting',
    'goal_per_shot': 'shooting',
    'goal_per_sot': 'shooting',
    'npxg_per_shot': 'shooting',
    'avg_shot_dist': 'shooting',
    'non_pen_shots': 'shooting',
    'non_pen_goal_per_non_pen_shot': 'shooting'}
