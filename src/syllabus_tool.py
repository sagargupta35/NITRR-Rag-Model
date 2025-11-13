from langchain.tools import tool
from pydantic import BaseModel 
from typing import List, Literal
from typing import Optional
from src.fs_utils import get_root_dir
from pathlib import Path

import json
import sqlite3

class Unit(BaseModel):
    unit_number: int
    unit_name: str
    topics: List[str]

_valid_departments = Literal[
    'cse', 'it', 'electronics', 'electrical', 'mechanical', 
    'meta', 'civil', 'chemical', 'biotech', 'biomed', 'mining'
]

class Subject(BaseModel):
    name: str
    semester: int
    department: _valid_departments
    credits: str
    status: str
    code: str
    pre_requisites: List[str]
    units: List[Unit]
    course_material: List[str]

@tool
def syllabus_tool(department: str, semester: Optional[int]) -> str:
    """
        Tool: syllabus_tool

        Purpose:
        Use this tool to retrieve detailed syllabus information for subjects at NIT Raipur. 
        This is the primary tool for any questions related to courses, subjects, syllabus content (units), 
        credits, pre-requisites, or course materials.

        Arguments:

        1. department (str):
        REQUIRED. The single, valid department code. You must infer this from the user's query 
        (e.g., if the user asks about "computer science", you must use the code "cse").
        
        Valid department codes are:
        ['cse', 'it', 'electronics', 'electrical', 'mechanical', 'meta', 'civil', 'chemical', 'biotech', 'biomed', 'mining']

        2. semester (int):
        OPTIONAL. The semester number. Must be an integer between 1 and 8.
        
        Agent Strategy Hint:

        * You MUST always provide the `department` argument.
        * Only provide the `semester` argument if the user *explicitly* asks for a specific semester.
        * If the `semester` argument is omitted, the tool will return subjects for *all* semesters (1-8) for that department.
        * This tool can only search ONE department at a time. If the user's query involves more 
            than one department (e.g., "List subjects for cse and it"), you MUST call this tool 
            multiple times (once for 'cse' and once for 'it').

        Returns:
        A JSON string representing a list of matching `Subject` objects. An empty list '[]' is 
        returned if no matches are found.
    """

    if department not in _valid_departments.__args__:
        return f"department should be one of {_valid_departments}"
    
    if semester and (semester <= 0 or semester > 8):
        return f"semester should be between 1 and 8 inclusive"

    # create a cursor to talk with db
    db_path: Path = get_root_dir() / "data" / "syllabus.db"
    if not db_path.is_file():
        return "database not found. cannot retrieve syllabus"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # form the sql query
    sql_query = f'SELECT * FROM subjects WHERE department=?'
    params: list = [department]
    if semester:
        sql_query += " AND semester=?"
        params.append(semester)

    # get the rows from db
    rows = cur.execute(sql_query, tuple(params))

    # convert rows to subjects
    subjects = []
    for row in rows:
        subject = _convert_row_to_subject(row)
        subjects.append(subject)

    # close the connection
    conn.close()

    subjects_as_dicts = [s.model_dump() for s in subjects]
    return json.dumps(subjects_as_dicts)

    
def _convert_row_to_subject(row) -> Subject:
    _, name, semester, department, credits, status, code, pre, units, material = row

    subject = Subject(
        name=name,
        semester=semester,
        department=department,
        credits=credits,
        status=status,
        code=code,
        pre_requisites=json.loads(pre),
        units=[Unit(**u) for u in json.loads(units)],
        course_material=json.loads(material)
    )

    return subject

if __name__ == "__main__":
    while True:
        department = input(">> Enter the departement: ")
        semester = int(input(">> Enter the semester: "))
        if semester == -1:
            semester = None

        print(syllabus_tool.invoke({'department': department, 'semester': semester}))
        
