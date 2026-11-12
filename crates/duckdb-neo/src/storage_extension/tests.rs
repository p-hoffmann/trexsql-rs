use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    DuckDBType, FromValue, Parameters, Result, ToValue,
    connection::{Connection, Context},
    environment::{Environment, StorageLocation},
    error::Error,
    storage_extension::{AttachInfo, NameList, StorageExtensionBuilder, StorageExtensionCallbacks, TableBind},
};

#[derive(Default)]
struct Calls {
    list_schemas: AtomicUsize,
    list_tables: AtomicUsize,
    bind: AtomicUsize,
    dropped: AtomicUsize,
    options: Mutex<Vec<(String, String)>>,
}

/// Schemas `Sales` (Orders, Items, Ghost, Broken) and `hr` (people).
struct FakeRemote {
    calls: Arc<Calls>,
}

struct FakeCatalog {
    label: String,
    calls: Arc<Calls>,
}

impl Drop for FakeCatalog {
    fn drop(&mut self) {
        self.calls.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl StorageExtensionCallbacks for FakeRemote {
    type Catalog = FakeCatalog;

    fn attach(&self, _context: &Context, info: &AttachInfo<'_>) -> Result<FakeCatalog> {
        if info.path()? == "refuse" {
            return Err(Error::invalid_input("the remote refused the connection".to_owned()));
        }
        let mut options = Vec::new();
        for (name, value) in info.options()? {
            options.push((name, String::from_value(&value)?.unwrap_or_default()));
        }
        *self.calls.options.lock().unwrap() = options;
        if let Some(schema) = info.option("SCHEMA")? {
            info.set_default_schema(&String::from_value(&schema)?.unwrap_or_default())?;
        }
        Ok(FakeCatalog {
            label: format!("{}@{}", info.name()?, info.path()?),
            calls: self.calls.clone(),
        })
    }

    fn list_schemas(&self, _context: &Context, _catalog: &FakeCatalog, names: &NameList<'_>) -> Result<()> {
        self.calls.list_schemas.fetch_add(1, Ordering::SeqCst);
        names.add("Sales")?;
        names.add("hr")
    }

    fn list_tables(
        &self,
        _context: &Context,
        _catalog: &FakeCatalog,
        schema: &str,
        names: &NameList<'_>,
    ) -> Result<()> {
        self.calls.list_tables.fetch_add(1, Ordering::SeqCst);
        let tables: &[&str] = match schema {
            "Sales" => &["Orders", "Items", "Ghost", "Broken"],
            "hr" => &["people"],
            _ => &[],
        };
        tables.iter().try_for_each(|table| names.add(table))
    }

    fn bind_table(
        &self,
        context: &Context,
        catalog: &FakeCatalog,
        schema: &str,
        table: &str,
        bind: &TableBind<'_>,
    ) -> Result<()> {
        self.calls.bind.fetch_add(1, Ordering::SeqCst);
        match (schema, table) {
            ("Sales", "Orders") => {
                // The catalog state reaches the bind: the repeated value is the database's label.
                bind.set_function_name("repeat")?;
                bind.add_parameter(catalog.label.as_str().value(context)?)?;
                bind.add_parameter(3i64.value(context)?)?;
            }
            ("Sales", "Items") => {
                bind.set_function_name("repeat")?;
                bind.add_parameter(7i64.value(context)?)?;
                bind.add_parameter(2i64.value(context)?)?;
                bind.add_column("item_id", &i64::logical_type(context)?)?;
            }
            ("Sales", "Broken") => return Err(Error::invalid_input("the remote table is unreadable".to_owned())),
            ("hr", "people") => {
                bind.set_function_name("test_all_types")?;
                bind.add_parameter_with_name("use_large_enum", true.value(context)?)?;
            }
            _ => {}
        }
        Ok(())
    }
}

fn single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut query = conn.query(sql, Parameters::None)?;
    let chunk = query.next_chunk()?.expect("one row");
    Ok(*chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?.expect("non-NULL"))
}

fn single_string(conn: &Connection, sql: &str) -> Result<String> {
    let mut query = conn.query(sql, Parameters::None)?;
    let chunk = query.next_chunk()?.expect("one row");
    let vector = chunk.get_vector_at::<String>(0)?;
    let value = vector.reader()?.get(0)?.expect("non-NULL").to_string();
    Ok(value)
}

fn error_text(conn: &Connection, sql: &str) -> String {
    match conn
        .query(sql, Parameters::None)
        .and_then(|mut q| q.next_chunk().map(|_| ()))
    {
        Ok(()) => "<no error>".to_owned(),
        Err(error) => error.to_string(),
    }
}

#[test]
fn test_storage_extension_attach_and_query() -> Result<()> {
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let calls = Arc::new(Calls::default());
    StorageExtensionBuilder::new(FakeRemote { calls: calls.clone() }).register(&instance, "fake")?;
    let conn = instance.connect()?;

    conn.execute(
        "ATTACH 'remote' AS r (TYPE fake, SCHEMA 'Sales', TOKEN 'secret')",
        Parameters::None,
    )?;
    assert_eq!(
        *calls.options.lock().unwrap(),
        vec![
            ("schema".to_owned(), "Sales".to_owned()),
            ("token".to_owned(), "secret".to_owned())
        ]
    );
    assert_eq!(calls.list_schemas.load(Ordering::SeqCst), 0, "listing is lazy");

    assert_eq!(
        single_string(&conn, "SELECT * FROM r.Sales.Orders LIMIT 1")?,
        "r@remote"
    );
    assert_eq!(single_i64(&conn, "SELECT count(*) FROM r.sales.orders")?, 3);
    // The SCHEMA option made Sales the default schema.
    assert_eq!(single_i64(&conn, "SELECT count(*) FROM r.Orders")?, 3);
    assert_eq!(single_i64(&conn, "SELECT sum(item_id)::BIGINT FROM r.Sales.Items")?, 14);
    assert_eq!(single_i64(&conn, "SELECT count(*) FROM r.hr.people")?, 3);

    assert_eq!(calls.list_schemas.load(Ordering::SeqCst), 1);
    assert_eq!(calls.list_tables.load(Ordering::SeqCst), 2);
    assert_eq!(calls.bind.load(Ordering::SeqCst), 3, "each table is bound once");

    assert!(error_text(&conn, "SELECT * FROM r.Sales.Ghost").contains("Ghost"));
    assert!(error_text(&conn, "SELECT * FROM r.Sales.Broken").contains("the remote table is unreadable"));
    assert!(error_text(&conn, "CREATE TABLE r.Sales.t (i INTEGER)").contains("read-only"));

    assert_eq!(calls.dropped.load(Ordering::SeqCst), 0);
    conn.execute("DETACH r", Parameters::None)?;
    assert_eq!(
        calls.dropped.load(Ordering::SeqCst),
        1,
        "DETACH drops the catalog state"
    );

    assert!(error_text(&conn, "ATTACH 'refuse' AS r (TYPE fake)").contains("the remote refused the connection"));
    Ok(())
}

#[test]
fn test_storage_extension_type_name_taken() -> Result<()> {
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let calls = Arc::new(Calls::default());
    StorageExtensionBuilder::new(FakeRemote { calls: calls.clone() }).register(&instance, "fake")?;
    let again = StorageExtensionBuilder::new(FakeRemote { calls }).register(&instance, "FAKE");
    assert!(again.is_err());
    Ok(())
}
