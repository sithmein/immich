import 'dart:math' as math;

import 'package:collection/collection.dart';
import 'package:immich_mobile/domain/interfaces/asset.interface.dart';
import 'package:immich_mobile/domain/models/asset.model.dart';
import 'package:immich_mobile/domain/models/render_list_element.model.dart';
import 'package:immich_mobile/service_locator.dart';

class RenderList {
  final List<RenderListElement> elements;
  final int totalCount;

  /// global offset of assets in [_buf]
  int _bufOffset = 0;

  /// reference to batch of assets loaded from DB with offset [_bufOffset]
  List<Asset> _buf = [];

  RenderList({required this.elements, required this.totalCount});

  /// Loads the requested assets from the database to an internal buffer if not cached
  /// and returns a slice of that buffer
  Future<List<Asset>> loadAssets(int offset, int count) async {
    assert(offset >= 0);
    assert(count > 0);
    assert(offset + count <= totalCount);

    // general case: we have the query to load assets via offset from the DB on demand
    if (offset < _bufOffset || offset + count > _bufOffset + _buf.length) {
      // the requested slice (offset:offset+count) is not contained in the cache buffer `_buf`
      // thus, fill the buffer with a new batch of assets that at least contains the requested
      // assets and some more

      final bool forward = _bufOffset < offset;
      // if the requested offset is greater than the cached offset, the user scrolls forward "down"
      const batchSize = 256;
      const oppositeSize = 64;

      // make sure to load a meaningful amount of data (and not only the requested slice)
      // otherwise, each call to [loadAssets] would result in DB call trashing performance
      // fills small requests to [batchSize], adds some legroom into the opposite scroll direction for large requests
      final len = math.max(batchSize, count + oppositeSize);
      // when scrolling forward, start shortly before the requested offset...
      // when scrolling backward, end shortly after the requested offset...
      // ... to guard against the user scrolling in the other direction
      // a tiny bit resulting in a another required load from the DB
      final start = math.max(
        0,
        forward
            ? offset - oppositeSize
            : (len > batchSize ? offset : offset + count - len),
      );
      // load the calculated batch (start:start+len) from the DB and put it into the buffer
      _buf =
          await di<IAssetRepository>().fetchAssets(offset: start, limit: len);
      _bufOffset = start;

      assert(_bufOffset <= offset);
      assert(_bufOffset + _buf.length >= offset + count);
    }
    // return the requested slice from the buffer (we made sure before that the assets are loaded!)
    return _buf.slice(offset - _bufOffset, offset - _bufOffset + count);
  }
}
