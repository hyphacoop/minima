import logging
import time
from pathlib import Path
from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from singleton import Singleton
from enum import Enum

logger = logging.getLogger(__name__)


class IndexingStatus(Enum):
    new_file = 1
    need_reindexing = 2
    no_need_reindexing = 3
    failed = 4


class MinimaDoc(SQLModel, table=True):
    fpath: str = Field(primary_key=True)
    last_updated_seconds: int | None = Field(default=None, index=True)
    description: str = Field(default="")
    tags: str = Field(default="[]")
    metadata_updated_seconds: int | None = Field(default=None, index=True)


class MinimaDocUpdate(SQLModel):
    fpath: str | None = None
    last_updated_seconds: int | None = None
    description: str | None = None
    tags: str | None = None
    metadata_updated_seconds: int | None = None


sqlite_file_name = "/indexer/storage/database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


class MinimaStore(metaclass=Singleton):

    @staticmethod
    def create_db_and_tables():
        SQLModel.metadata.create_all(engine)
        MinimaStore._migrate_minima_doc_columns()

    @staticmethod
    def _migrate_minima_doc_columns():
        """Add metadata columns when upgrading an existing SQLite database."""
        with Session(engine) as session:
            rows = session.exec(text("PRAGMA table_info(minimadoc)")).all()
            existing_columns = {row[1] for row in rows}
            migrations = {
                "description": "ALTER TABLE minimadoc ADD COLUMN description TEXT NOT NULL DEFAULT ''",
                "tags": "ALTER TABLE minimadoc ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
                "metadata_updated_seconds": "ALTER TABLE minimadoc ADD COLUMN metadata_updated_seconds INTEGER",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    session.exec(text(statement))
            session.commit()

    @staticmethod
    def delete_m_doc(fpath: str) -> None:
        with Session(engine) as session:
            statement = select(MinimaDoc).where(MinimaDoc.fpath == fpath)
            results = session.exec(statement)
            doc = results.one()
            session.delete(doc)
            session.commit()
            print("doc deleted:", doc)

    @staticmethod
    def select_m_doc(fpath: str) -> MinimaDoc:
        with Session(engine) as session:
            statement = select(MinimaDoc).where(MinimaDoc.fpath == fpath)
            results = session.exec(statement)
            doc = results.one()
            print("doc:", doc)
            return doc

    @staticmethod
    def select_all_indexed_paths() -> list[str]:
        """Return all file paths currently indexed in the database"""
        with Session(engine) as session:
            statement = select(MinimaDoc.fpath)
            results = session.exec(statement)
            return [fpath for fpath in results]

    @staticmethod
    def list_docs() -> list[MinimaDoc]:
        with Session(engine) as session:
            statement = select(MinimaDoc).order_by(MinimaDoc.fpath)
            return list(session.exec(statement))

    @staticmethod
    def find_by_filename(filename: str) -> list[str]:
        """Find all file paths in the database with the given basename."""
        with Session(engine) as session:
            docs = session.exec(select(MinimaDoc)).all()
            return [doc.fpath for doc in docs if Path(doc.fpath).name == filename]

    @staticmethod
    def upsert_metadata(
        fpath: str,
        description: str,
        tags: str,
        last_updated_seconds: int | None = None,
    ) -> MinimaDoc:
        with Session(engine) as session:
            statement = select(MinimaDoc).where(MinimaDoc.fpath == fpath)
            doc = session.exec(statement).first()
            metadata_updated_seconds = max(round(time.time()), (doc.last_updated_seconds or 0) + 1) if doc else round(time.time())
            if doc is None:
                doc = MinimaDoc(
                    fpath=fpath,
                    last_updated_seconds=last_updated_seconds,
                    description=description,
                    tags=tags,
                    metadata_updated_seconds=metadata_updated_seconds,
                )
            else:
                doc_update = MinimaDocUpdate(
                    description=description,
                    tags=tags,
                    metadata_updated_seconds=metadata_updated_seconds,
                )
                if last_updated_seconds is not None:
                    doc_update.last_updated_seconds = last_updated_seconds
                doc.sqlmodel_update(doc_update.model_dump(exclude_unset=True))
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    @staticmethod
    def find_removed_files(existing_file_paths: set[str]):
        removed_files: list[str] = []
        with Session(engine) as session:
            statement = select(MinimaDoc)
            results = session.exec(statement)
            logger.debug(f"find_removed_files count found {results}")
            for doc in results:
                logger.debug(f"find_removed_files file {doc.fpath} checking to remove")
                if doc.fpath not in existing_file_paths:
                    logger.debug(f"find_removed_files file {doc.fpath} does not exist anymore, removing")
                    removed_files.append(doc.fpath)
        for fpath in removed_files:
            MinimaStore.delete_m_doc(fpath)
        return removed_files

    @staticmethod
    def check_needs_indexing(fpath: str, last_updated_seconds: int) -> IndexingStatus:
        indexing_status: IndexingStatus = IndexingStatus.no_need_reindexing
        try:
            with Session(engine) as session:
                statement = select(MinimaDoc).where(MinimaDoc.fpath == fpath)
                results = session.exec(statement)
                doc = results.first()
                if doc is not None:
                    logger.debug(
                        f"file {fpath} new last updated={last_updated_seconds} old last updated: {doc.last_updated_seconds}"
                    )
                    indexed_seconds = doc.last_updated_seconds or 0
                    metadata_seconds = doc.metadata_updated_seconds or 0
                    if indexed_seconds < last_updated_seconds or indexed_seconds < metadata_seconds:
                        indexing_status = IndexingStatus.need_reindexing
                        logger.debug(f"file {fpath} needs indexing, timestamp or metadata changed")
                        doc_update = MinimaDocUpdate(
                            fpath=fpath,
                            last_updated_seconds=max(last_updated_seconds, metadata_seconds),
                        )
                        doc_data = doc_update.model_dump(exclude_unset=True)
                        doc.sqlmodel_update(doc_data)
                        session.add(doc)
                        session.commit()
                    else:
                        logger.debug(f"file {fpath} doesn't need indexing, timestamp same")
                else:
                    doc = MinimaDoc(fpath=fpath, last_updated_seconds=last_updated_seconds)
                    session.add(doc)
                    session.commit()
                    logger.debug(f"file {fpath} needs indexing, new file")
                    indexing_status = IndexingStatus.new_file
            return indexing_status
        except Exception as e:
            logger.error(f"error updating file in the store {e}, skipping indexing")
            return IndexingStatus.no_need_reindexing
