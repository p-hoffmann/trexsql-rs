//! Storage extensions: `ATTACH '<path>' AS db (TYPE <type>)` backed by callbacks.
//!
//! A storage extension serves a read-only catalog whose schemas and tables come from a
//! [`StorageExtensionCallbacks`] implementation, which is the shape of a remote-database connector:
//!
//! - [`attach`](StorageExtensionCallbacks::attach) validates the path and options and returns per-database state,
//! - [`list_schemas`](StorageExtensionCallbacks::list_schemas) and
//!   [`list_tables`](StorageExtensionCallbacks::list_tables) report names,
//! - [`bind_table`](StorageExtensionCallbacks::bind_table) names the table function that reads one table.
//!
//! DuckDB caches every answer for as long as the database stays attached, lists lazily, and binds a table on first
//! use. A failed callback is not cached. The scan function must be a plain table function, not a table in-out
//! function such as `range`.

use crate::{
    Result,
    builder_helpers::{OpaqueHandle, get_opaque_data_ref, handle_unwind, into_opaque},
    check_api_call,
    connection::Context,
    ffi,
    ffi_str::DuckDBStr,
    handles::StorageExtensionBuilderLink,
    logical_type::LogicalType,
    qualified_name::QualifiedName,
    raw::{FromRaw, RawExt},
    value::Value,
};

/// The ATTACH being served.
#[derive(Debug)]
pub struct AttachInfo<'a> {
    info: ffi::duckdb_v2_storage_attach_info_handle,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl AttachInfo<'_> {
    /// The name the database is attached as.
    pub fn name(&self) -> Result<&str> {
        check_api_call!(ffi::duckdb_v2_storage_attach_info_get_name, self.info, RET)
            .map(|name| DuckDBStr::from_raw(name).as_str().unwrap_or_default())
    }

    /// The attached path, with a leading `<type>:` prefix already stripped when it selected the type.
    pub fn path(&self) -> Result<&str> {
        check_api_call!(ffi::duckdb_v2_storage_attach_info_get_path, self.info, RET)
            .map(|path| DuckDBStr::from_raw(path).as_str().unwrap_or_default())
    }

    /// The `(KEY value)` options passed on to the storage extension, lowercased and ordered by name.
    pub fn options(&self) -> Result<Vec<(String, Value)>> {
        let count = check_api_call!(ffi::duckdb_v2_storage_attach_info_get_option_count, self.info, RET)?;
        let mut options = Vec::with_capacity(count as usize);
        for index in 0..count {
            let name = check_api_call!(
                ffi::duckdb_v2_storage_attach_info_get_option_name,
                self.info,
                index,
                RET
            )?;
            let name = DuckDBStr::from_raw(name).as_str().unwrap_or_default().to_owned();
            let handle = check_api_call!(
                ffi::duckdb_v2_storage_attach_info_get_option_value,
                self.info,
                index,
                RET
            )?;
            options.push((name, Value { handle }));
        }
        Ok(options)
    }

    /// The value of the option `name` (matched case-insensitively), if it was given.
    pub fn option(&self, name: &str) -> Result<Option<Value>> {
        Ok(self
            .options()?
            .into_iter()
            .find(|(option, _)| option.eq_ignore_ascii_case(name))
            .map(|(_, value)| value))
    }

    /// Set the schema that `<database>.<table>` and `USE <database>` resolve to. Defaults to `main`.
    pub fn set_default_schema(&self, schema: &str) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_storage_attach_info_set_default_schema,
            self.info,
            &schema.into()
        )
    }
}

/// Collects the names reported by a list callback.
#[derive(Debug)]
pub struct NameList<'a> {
    info: ffi::duckdb_v2_storage_list_info_handle,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl NameList<'_> {
    /// Report one schema or table. Names match case-insensitively; a repeated name is kept once.
    pub fn add(&self, name: &str) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_storage_list_info_add_name, self.info, &name.into())
    }
}

/// Names the table function that reads one table.
#[derive(Debug)]
pub struct TableBind<'a> {
    info: ffi::duckdb_v2_storage_bind_info_handle,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl TableBind<'_> {
    /// Name the table function, optionally qualified by schema and catalog. Not naming one makes the table unknown.
    pub fn set_function(&self, name: &QualifiedName) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_storage_bind_info_set_function_name,
            self.info,
            name.raw()
        )
    }

    /// Name an unqualified table function.
    pub fn set_function_name(&self, name: &str) -> Result<()> {
        self.set_function(&QualifiedName::from_parts(&[name])?)
    }

    /// Append a positional argument. Requires the function to be named first.
    pub fn add_parameter(&self, value: Value) -> Result<()> {
        check_api_call!(ffi::duckdb_v2_storage_bind_info_add_argument, self.info, value.handle)
    }

    /// Append a named argument, as `name := value`. Requires the function to be named first.
    pub fn add_parameter_with_name(&self, name: &str, value: Value) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_storage_bind_info_add_named_argument,
            self.info,
            &name.into(),
            value.handle,
        )
    }

    /// Declare the next column. Optional: without declarations the columns are those the function binds to; with
    /// them, they must match the function's columns in order and type, and listing the table no longer binds it.
    pub fn add_column(&self, name: &str, logical_type: &LogicalType) -> Result<()> {
        check_api_call!(
            ffi::duckdb_v2_storage_bind_info_add_column,
            self.info,
            &name.into(),
            logical_type.handle,
        )
    }
}

/// The callbacks behind a storage type.
///
/// Callbacks may run on several connections at once. `context` is the context of the statement that needed the
/// answer; it can read settings but must not run queries on this database.
pub trait StorageExtensionCallbacks: Send + Sync + 'static {
    /// Per-database state, created by [`Self::attach`] and dropped on DETACH.
    type Catalog: Send + Sync + 'static;

    /// Serve an ATTACH: validate the path and options, and return the database's state. An error fails the ATTACH.
    fn attach(&self, context: &Context, info: &AttachInfo<'_>) -> Result<Self::Catalog>;

    /// Report the schemas of an attached database.
    fn list_schemas(&self, context: &Context, catalog: &Self::Catalog, names: &NameList<'_>) -> Result<()>;

    /// Report the tables of one schema.
    fn list_tables(&self, context: &Context, catalog: &Self::Catalog, schema: &str, names: &NameList<'_>)
    -> Result<()>;

    /// Name the table function that reads `schema.table`, or return without naming one to make it unknown.
    fn bind_table(
        &self,
        context: &Context,
        catalog: &Self::Catalog,
        schema: &str,
        table: &str,
        bind: &TableBind<'_>,
    ) -> Result<()>;
}

unsafe fn implementation<'a, T: StorageExtensionCallbacks>(user_data: *mut std::ffi::c_void) -> Result<&'a T> {
    unsafe { get_opaque_data_ref::<T>(user_data) }
        .ok_or_else(|| crate::error::Error::api_error("storage extension user data is missing".to_owned()))
}

unsafe fn catalog<'a, T: StorageExtensionCallbacks>(catalog_data: *mut std::ffi::c_void) -> Result<&'a T::Catalog> {
    unsafe { get_opaque_data_ref::<T::Catalog>(catalog_data) }
        .ok_or_else(|| crate::error::Error::api_error("storage extension catalog data is missing".to_owned()))
}

unsafe extern "C" fn attach_callback<T: StorageExtensionCallbacks>(
    info: ffi::duckdb_v2_storage_attach_info_handle,
    context: ffi::duckdb_v2_context_handle,
    err: *mut ffi::duckdb_v2_error_info_handle,
) {
    handle_unwind(
        || {
            let user_data = check_api_call!(ffi::duckdb_v2_storage_attach_info_get_user_data, info, RET)?;
            let implementation = unsafe { implementation::<T>(user_data) }?;
            let attach = AttachInfo {
                info,
                _marker: std::marker::PhantomData,
            };
            let state = implementation.attach(&unsafe { Context::from_raw(context) }, &attach)?;
            check_api_call!(
                ffi::duckdb_v2_storage_attach_info_set_catalog_data,
                info,
                &mut into_opaque(state)
            )
        },
        err,
    );
}

unsafe extern "C" fn list_schemas_callback<T: StorageExtensionCallbacks>(
    info: ffi::duckdb_v2_storage_list_info_handle,
    context: ffi::duckdb_v2_context_handle,
    err: *mut ffi::duckdb_v2_error_info_handle,
) {
    handle_unwind(
        || {
            let user_data = check_api_call!(ffi::duckdb_v2_storage_list_info_get_user_data, info, RET)?;
            let catalog_data = check_api_call!(ffi::duckdb_v2_storage_list_info_get_catalog_data, info, RET)?;
            let names = NameList {
                info,
                _marker: std::marker::PhantomData,
            };
            unsafe { implementation::<T>(user_data) }?.list_schemas(
                &unsafe { Context::from_raw(context) },
                unsafe { catalog::<T>(catalog_data) }?,
                &names,
            )
        },
        err,
    );
}

unsafe extern "C" fn list_tables_callback<T: StorageExtensionCallbacks>(
    info: ffi::duckdb_v2_storage_list_info_handle,
    context: ffi::duckdb_v2_context_handle,
    err: *mut ffi::duckdb_v2_error_info_handle,
) {
    handle_unwind(
        || {
            let user_data = check_api_call!(ffi::duckdb_v2_storage_list_info_get_user_data, info, RET)?;
            let catalog_data = check_api_call!(ffi::duckdb_v2_storage_list_info_get_catalog_data, info, RET)?;
            let schema = check_api_call!(ffi::duckdb_v2_storage_list_info_get_schema_name, info, RET)?;
            let schema = DuckDBStr::from_raw(schema).as_str().unwrap_or_default();
            let names = NameList {
                info,
                _marker: std::marker::PhantomData,
            };
            unsafe { implementation::<T>(user_data) }?.list_tables(
                &unsafe { Context::from_raw(context) },
                unsafe { catalog::<T>(catalog_data) }?,
                schema,
                &names,
            )
        },
        err,
    );
}

unsafe extern "C" fn bind_table_callback<T: StorageExtensionCallbacks>(
    info: ffi::duckdb_v2_storage_bind_info_handle,
    context: ffi::duckdb_v2_context_handle,
    err: *mut ffi::duckdb_v2_error_info_handle,
) {
    handle_unwind(
        || {
            let user_data = check_api_call!(ffi::duckdb_v2_storage_bind_info_get_user_data, info, RET)?;
            let catalog_data = check_api_call!(ffi::duckdb_v2_storage_bind_info_get_catalog_data, info, RET)?;
            let schema = check_api_call!(ffi::duckdb_v2_storage_bind_info_get_schema_name, info, RET)?;
            let table = check_api_call!(ffi::duckdb_v2_storage_bind_info_get_table_name, info, RET)?;
            let bind = TableBind {
                info,
                _marker: std::marker::PhantomData,
            };
            unsafe { implementation::<T>(user_data) }?.bind_table(
                &unsafe { Context::from_raw(context) },
                unsafe { catalog::<T>(catalog_data) }?,
                DuckDBStr::from_raw(schema).as_str().unwrap_or_default(),
                DuckDBStr::from_raw(table).as_str().unwrap_or_default(),
                &bind,
            )
        },
        err,
    );
}

/// Registers a storage type backed by a [`StorageExtensionCallbacks`] implementation.
///
/// Register before issuing ATTACH statements. The type name is matched case-insensitively and must not be taken.
#[derive(Debug)]
pub struct StorageExtensionBuilder<T> {
    implementation: T,
}

impl<T: StorageExtensionCallbacks> StorageExtensionBuilder<T> {
    /// Create a builder from its callback implementation.
    pub fn new(implementation: T) -> Self {
        Self { implementation }
    }

    /// Register under `type_name` through an instance or a loading extension, consuming the builder.
    #[allow(private_bounds)]
    pub fn register<L: StorageExtensionBuilderLink>(self, link: &L, type_name: &str) -> Result<()> {
        let data = OpaqueHandle::new(self.implementation);
        let handle = link.create_storage_extension_handle()?;

        check_api_call!(
            ffi::duckdb_v2_storage_extension_set_attach,
            handle.raw(),
            Some(attach_callback::<T>)
        )?;
        check_api_call!(
            ffi::duckdb_v2_storage_extension_set_list_schemas,
            handle.raw(),
            Some(list_schemas_callback::<T>)
        )?;
        check_api_call!(
            ffi::duckdb_v2_storage_extension_set_list_tables,
            handle.raw(),
            Some(list_tables_callback::<T>)
        )?;
        check_api_call!(
            ffi::duckdb_v2_storage_extension_set_bind_table,
            handle.raw(),
            Some(bind_table_callback::<T>)
        )?;
        check_api_call!(
            ffi::duckdb_v2_storage_extension_set_user_data,
            handle.raw(),
            &mut data.to_handle()
        )?;

        link.register_storage_extension_handle(handle, type_name)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
