import 'package:flutter/material.dart';
import 'package:immich_mobile/domain/interfaces/store.interface.dart';
import 'package:immich_mobile/domain/models/user.model.dart';
import 'package:immich_mobile/domain/utils/store_converters.dart';
import 'package:immich_mobile/presentation/theme/app_theme.dart';

@immutable
class StoreValue<T> {
  final int id;
  final T? value;

  const StoreValue({required this.id, this.value});

  @override
  bool operator ==(covariant StoreValue other) {
    if (identical(this, other)) return true;

    return other.hashCode == hashCode;
  }

  @override
  int get hashCode => id.hashCode ^ value.hashCode;
}

class StoreKeyNotFoundException implements Exception {
  final StoreKey key;
  const StoreKeyNotFoundException(this.key);

  @override
  String toString() => "Key '${key.name}' not found in Store";
}

/// Key for each possible value in the `Store`.
/// Also stores the converter to convert the value to and from the store and the type of value stored in the Store
enum StoreKey<T, U> {
  serverEndpoint<String, String>._(
    0,
    converter: StoreStringConverter(),
    type: String,
  ),
  accessToken<String, String>._(
    1,
    converter: StoreStringConverter(),
    type: String,
  ),
  currentUser<User, String>._(
    2,
    converter: StoreUserConverter(),
    type: String,
  ),
  // App settings
  appTheme<AppTheme, int>._(
    1000,
    converter: StoreEnumConverter(AppTheme.values),
    type: int,
  ),
  themeMode<ThemeMode, int>._(
    1001,
    converter: StoreEnumConverter(ThemeMode.values),
    type: int,
  ),
  darkMode<bool, int>._(1002, converter: StoreBooleanConverter(), type: int);

  const StoreKey._(this.id, {required this.converter, required this.type});
  final int id;

  /// Primitive Type is also stored here to easily fetch it during runtime
  final Type type;
  final IStoreConverter<T, U> converter;
}
