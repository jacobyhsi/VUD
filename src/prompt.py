import pandas as pd
from typing import Optional

PROMPT_TYPES_TO_TEXT_TEMPLATE = {
    "tabular": \
"""{icl}
{note} <output>""",
    "toy_classification": \
"""{icl}
 {note} <output>""",
    "toy_regression": \
"""{icl}
 {note} <output>""",
    "bandit_classification": \
"""{icl}
 {note} <reward>""",
    "bandit_regression": \
"""{icl}
 {note} <reward>""",
}



class Prompt():
    def __init__(self, prompt_type = "tabular") -> None:
        self.prompt_type = prompt_type

    @property
    def prompt_text(self) -> str:
        return PROMPT_TYPES_TO_TEXT_TEMPLATE[self.prompt_type]
            
    def get_puzD_prompt(self, z, D):
        return self.prompt_text.format(self=self, note=z, icl=D)

    def get_pyxuzD_prompt(self, x, icl):
        return self.prompt_text.format(self=self, note=x, icl=icl)

    def get_pyxD_prompt(self, x, D):
        return self.prompt_text.format(self=self, note=x, icl=D)
    
class ToyPrompt(Prompt):
    def __init__(self, prompt_type) -> None:
        super().__init__(prompt_type=prompt_type)
    
    def note_label_prompt(self, note: str, label: str) -> str:
        prompt = f""" {note} <output>{label}</output>"""
        
        return prompt
    
    def note_label_df_to_icl_string(
            self,
            note_label_df: pd.DataFrame,
            permutation_seed: int,
            z_note: Optional[str] = None,
            u_label: Optional[str|int|float] = None,
        ):
        """
        Converts a DataFrame of notes and labels to incontext examples for LLM.
        Shuffles the DataFrame before converting.
        
        If z_note and u_label are provided, the z_note and u_label will be added to data as well.
        """
        
        if z_note is not None and u_label is not None:
            z_note_label_df = pd.DataFrame([{"note": z_note, "label": u_label}])
            note_label_df = pd.concat([note_label_df, z_note_label_df], ignore_index=True)
            
        note_label_df = note_label_df.sample(frac=1, random_state=permutation_seed).reset_index(drop=True)
        
        in_context_examples = []
        
        for _, row in note_label_df.iterrows():
            in_context_examples.append(self.note_label_prompt(row['note'], row['label']))
        
        return "\n".join(in_context_examples)
    
    def get_general_prompt(
            self,
            D_df: pd.DataFrame,
            query_note: str,
            permutation_seed: int,
            icl_z_note: Optional[str] = None,
            icl_u_label: Optional[str|int|float] = None,
        ):
        """
        Returns a general prompt for the toy classification task.
        """
        
        icl_string = self.note_label_df_to_icl_string(
            D_df,
            permutation_seed,
            icl_z_note,
            icl_u_label,
        )
        
        return self.prompt_text.format(self=self, note=query_note, icl=icl_string)
    
class ToyClassificationPrompt(ToyPrompt):    
    def __init__(self) -> None:
        super().__init__(prompt_type="toy_classification")
    
    def note_label_prompt(self, note: str, label: str):
        prompt = f""" {note} <output>{label}</output>"""
        
        return prompt
    
class ToyRegressionPrompt(ToyPrompt):
    def __init__(self) -> None:
        super().__init__(prompt_type="toy_regression")
    
    def note_label_prompt(self, note: str, label: str):
        prompt = f""" {note} <output> {label} </output>"""
        
        return prompt
    
class BanditClassificationPrompt(ToyPrompt):
    def __init__(self) -> None:
        super().__init__(prompt_type="bandit_classification")
    
    def note_label_prompt(self, note: str, label: str):
        prompt = f""" {note} <reward>{label}</reward>"""
        
        return prompt
    
class BanditRegressionPrompt(ToyPrompt):
    def __init__(self) -> None:
        super().__init__(prompt_type="bandit_regression")
        
    def note_label_df_to_icl_string(
            self,
            note_label_df: pd.DataFrame,
            permutation_seed: int,
            z_note: Optional[str] = None,
            u_label: Optional[str|int|float] = None,
            decimal_places: int = 2,
        ):
        """
        Converts a DataFrame of notes and labels to incontext examples for LLM.
        Shuffles the DataFrame before converting.
        
        If z_note and u_label are provided, the z_note and u_label will be added to data as well.
        """
        
        if z_note is not None and u_label is not None:
            z_note_label_df = pd.DataFrame([{"note": z_note, "label": u_label}])
            note_label_df = pd.concat([note_label_df, z_note_label_df], ignore_index=True)
            
        note_label_df = note_label_df.sample(frac=1, random_state=permutation_seed).reset_index(drop=True)
        
        in_context_examples = []
        
        for _, row in note_label_df.iterrows():
            in_context_examples.append(self.note_label_prompt(row['note'], row['label'], decimal_places=decimal_places))
        
        return "\n".join(in_context_examples)
    
    def note_label_prompt(self, note: str, label: str, decimal_places: int = 2):
        prompt = f""" {note} <reward> {label:.{decimal_places}f} </reward>"""
        
        return prompt