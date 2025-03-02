from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
import joblib
import numpy as np
import os