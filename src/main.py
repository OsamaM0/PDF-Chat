from fastapi import FastAPI
import uvicorn
from routes import base, data, nlp, voice, document
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from helpers.logging_config import setup_logger
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

# Setup logger
logger = setup_logger()

app = FastAPI()

async def startup_span():
    """Start up the application and connect to the database and other services
    """
    # get settings
    settings = get_settings()
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
    
    # connect to the database with retry logic
    max_retries = 10
    retry_delay = 5  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            app.mongo_conn = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True,
                retryReads=True
            )
            # Force a command to check the connection
            await app.mongo_conn.admin.command('ping')
            app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
            logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DATABASE}")
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {e}")
                logger.error(f"Please ensure MongoDB is running at {settings.MONGODB_URL}")
                logger.error("The application will start but database functionality will be limited")
                # Don't raise - allow app to start with limited functionality
            logger.warning(f"Connection attempt {attempt} failed: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    # create the provider factories
    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # generation client
    logger.info(f"Initializing generation client with backend: {settings.GENERATION_BACKEND}")
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
    logger.info(f"Initializing embedding client with backend: {settings.EMBEDDING_BACKEND}")
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # vector db client
    logger.info(f"Initializing vector database with backend: {settings.VECTOR_DB_BACKEND}")
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    try:
        app.vectordb_client.connect()
        logger.info("Successfully connected to vector database")
    except Exception as e:
        logger.error(f"Failed to connect to vector database: {e}")
        logger.error("The application will start but vector database functionality will be limited")
        # Don't raise - allow app to start with limited functionality

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )
    logger.info("Application startup complete")


async def shutdown_span():
    """Shut down the application and close the database connection
    """
    logger.info("Shutting down application")
    try:
        if hasattr(app, 'mongo_conn'):
            app.mongo_conn.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
    
    try:
        if hasattr(app, 'vectordb_client'):
            app.vectordb_client.disconnect()
            logger.info("Vector database connection closed")
    except Exception as e:
        logger.error(f"Error closing vector database connection: {e}")
    
    logger.info("Application shutdown complete")

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
# app.include_router(document.voice_router)
app.include_router(voice.voice_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="localhost", port=8000)

# uvicorn main:app --host localhost --port 8000 --reload
