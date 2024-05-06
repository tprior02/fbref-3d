import pandas as pd
from typing import List
from maps import category_to_table, competitions_dictionary
import numpy as np
from copy import deepcopy

defense = pd.read_csv('data/defense.csv', index_col=0)
gca = pd.read_csv('data/gca.csv', index_col=0)
keepers = pd.read_csv('data/keepers.csv', index_col=0)
keepersadv = pd.read_csv('data/keepersadv.csv', index_col=0)
misc = pd.read_csv('data/misc.csv', index_col=0)
passing = pd.read_csv('data/passing.csv', index_col=0)
passing_types = pd.read_csv('data/passing_types.csv', index_col=0)
playingtime = pd.read_csv('data/playingtime.csv', index_col=0)
possession = pd.read_csv('data/possession.csv', index_col=0)
shooting = pd.read_csv('data/shooting.csv', index_col=0)
playerinfo = pd.read_csv('data/playerinfo.csv', index_col=0)
aggregates = pd.read_csv('data/aggregates.csv', index_col=0)

max_minutes = int(playingtime[['minutes']].groupby(level=0)['minutes'].sum().max())
max_value = float(playerinfo['value'].max())
min_age = int(playerinfo['age'].min())
max_age = int(playerinfo['age'].max())
min_height = float(playerinfo['height'].min())
max_height = float(playerinfo['height'].max())
club_options = playerinfo[playerinfo['value'] > 20]['current_club'].sort_values().unique().astype(str).tolist()
nation_options = playerinfo['citizenship'].sort_values().unique().astype(str).tolist()
name_options = playerinfo['simple_name'].sort_values().unique().astype(str).tolist()
agent_options = playerinfo['player_agent'].sort_values().unique().astype(str).tolist()
outfitter_options = playerinfo['outfitter'].sort_values().unique().astype(str).tolist()
initial_competitions = list(competitions_dictionary.values())
initial_competitions.remove('EL')


def single_table_query(
        category: str,
        seasons: List[int],
        competitions: List[str],
        label: str,
        group_by_season: bool,
) -> pd.DataFrame:
    """
    :param category: the category we want to query
    :param seasons: the seasons we want to query
    :param competitions: the competitions we want to query
    :param label: the label we will rename the category to
    :param group_by_season: should we group by season?
    :return: the category summed for the seasons and competitions provided
    """
    df = get_category_dataframe(category_to_table[category])
    df = df[(df['season'].isin(seasons) & df['comp'].isin(competitions))]
    if group_by_season:
        df = df[[category]]
        df = df.rename(columns={category: label})

        df = df.groupby(level=0).sum()
        return df.fillna(0)
    else:
        df.index = np.vectorize(lambda index, season: f'{index}{season}')(df.index, df['season'])
        df = df.groupby(level=0).sum()
        df = df[[category]]
        df = df.rename(columns={category: label})
        return df.fillna(0)


def multi_table_query(
        numerator_category: str,
        denominator_category: str,
        multiple: int,
        seasons: List[int],
        competitions: List[str],
        label: str,
        group_by_season: bool,
) -> pd.DataFrame:
    """
    :param numerator_category: the category of the numerator
    :param denominator_category: the category of the denominator
    :param multiple: the value to multiply the division by (100 for percentages, 1 otherwise)
    :param seasons: the seasons we want to query
    :param competitions: the competitions we want to query
    :param label: the label we will rename the category to
    :param group_by_season: should we group by season?

    :return: the sum of the numerator divided by the sum of the denominator multiplied by the multiple
    """
    df = single_table_query(denominator_category, seasons, competitions, 'denominator', group_by_season).join(
        single_table_query(numerator_category, seasons, competitions, 'numerator', group_by_season)
    ).fillna(0)
    df[label] = df['numerator'] / df['denominator'] * multiple
    return df[[label]]


def query(
        category: str,
        seasons: List[int],
        competitions: List[str],
        label: str,
        group_by_season: bool,
) -> pd.DataFrame:
    """
    This function either calls a single table query or a multi table query, depending on the category.

    :param category: the category we want to query
    :param seasons: the seasons we want to query
    :param competitions: the competitions we want to query
    :param label: the label we will rename the category to
    :param group_by_season: should we group by season?
    :return: the category summed for the seasons and competitions provided
    """
    if category in aggregates.index:
        return multi_table_query(
            aggregates.loc[category, 'numerator'],
            aggregates.loc[category, 'denominator'],
            aggregates.loc[category, 'multiple'],
            seasons,
            competitions,
            label,
            group_by_season
        )
    else:
        return single_table_query(
            category,
            seasons,
            competitions,
            label,
            group_by_season
        )


def minutes_query(competitions: List[str], seasons: List[int], group_by_season: bool) -> pd.DataFrame:
    """
    :param seasons: the seasons we want to query
    :param competitions: the competitions we want to query
    :param group_by_season: should we group by season?
    :return: the minutes summed for the seasons and competitions provided
    """
    df = playingtime
    df = df[(playingtime['season'].isin(seasons) & df['comp'].isin(competitions))]
    if group_by_season:
        df = df[['minutes']]
        df = df.groupby(level=0).sum()
        return df
    else:
        df.index = np.vectorize(lambda index, season: f'{index}{season}')(df.index, df['season'])
        df = df.groupby(level=0).sum()
        df = df[['minutes']]
        return df


def players_query(
        chosen_positions: List[str],
        chosen_values: List[int],
        chosen_ages: List[int],
        chosen_heights: List[int],
        chosen_clubs: List[str],
        chosen_nationalities: List[str],
        chosen_agents: List[str],
        chosen_outfitters: List[str],
        chosen_feet: List[str],
        additional_names_to_include: List[str],
        group_by_season: bool,
) -> pd.DataFrame:
    """
    :param chosen_positions: the list of positions to include
    :param chosen_values: the range of values to include (0 index=minimum, 1 index=maximum)
    :param chosen_ages:  the range of ages to include (0 index=minimum, 1 index=maximum)
    :param chosen_heights: the range of heights to include (0 index=minimum, 1 index=maximum)
    :param chosen_clubs: the list of clubs to include (ignored if a blank list provided)
    :param chosen_nationalities: the list of nationalities to include  (ignored if a blank list provided)
    :param chosen_agents: the list of agents to include  (ignored if a blank list provided)
    :param chosen_outfitters: the list of outfitters to include  (ignored if a blank list provided)
    :param chosen_feet: the list of feet to include
    :param additional_names_to_include: additional names we want to include on the plot, even if not in above criteria
    :param group_by_season: should we group by season?
    :return: player information for the provided criteria
    """

    df = playerinfo
    df = df[
        (df['current_club'].isin(chosen_clubs) if chosen_clubs != [] else True) &
        (df['citizenship'].isin(chosen_nationalities) if chosen_nationalities != [] else True) &
        (df['player_agent'].isin(chosen_agents) if chosen_agents != [] else True) &
        (df['outfitter'].isin(chosen_outfitters) if chosen_outfitters != [] else True) &
        (df['position'].isin(chosen_positions) if chosen_positions != [] else True) &
        (df['foot'].isin(chosen_feet)) &
        (df['value'] >= chosen_values[0]) &
        (df['value'] <= chosen_values[1]) &
        (df['height'] >= chosen_heights[0]) &
        (df['height'] <= chosen_heights[1]) &
        (df['age'] >= chosen_ages[0]) &
        (df['age'] <= chosen_ages[1]) |
        (df['simple_name'].isin(additional_names_to_include) if additional_names_to_include != [] else False)
        ]
    if not group_by_season:
        dfs = []
        for season in range(18, 25):
            df_season = deepcopy(df)
            df_season.index = np.vectorize(lambda index: f'{index}{season}')(df_season.index)
            df_season['last_name'] = np.vectorize(lambda name: f'{name} ({season-1}/{season})')(df_season['last_name'])
            df_season['name'] = np.vectorize(lambda name: f'{name} ({season-1}/{season})')(df_season['name'])
            dfs.append(df_season)
        df = pd.concat(dfs)
    return df


def get_category_dataframe(category: str) -> pd.DataFrame:
    match category:
        case 'keepers':
            df = keepers
        case 'keepersadv':
            df = keepersadv
        case 'shooting':
            df = shooting
        case 'passing':
            df = passing
        case 'passing_types':
            df = passing_types
        case 'gca':
            df = gca
        case 'defense':
            df = defense
        case 'possession':
            df = possession
        case 'playingtime':
            df = playingtime
        case 'misc':
            df = misc
        case _:
            return pd.DataFrame()
    return df
