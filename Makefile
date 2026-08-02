.PHONY: requirements check-requirements

# uv'nin ürettiği lock'ı saf requirements formatında dışa aktarır
requirements:
	uv export --frozen --no-hashes --format requirements-txt > requirements.txt

# CI/CD ve local commit hook'ları için drift kontrolü.
# NOT: `requirements` hedefine bağımlı DEĞİL — yerel çalışma ağacını sessizce
# değiştirmemesi için export /tmp'e yazılıp diff'lenir (Y4).
check-requirements:
	uv export --frozen --no-hashes --format requirements-txt > /tmp/req.check
	diff -u requirements.txt /tmp/req.check || \
		(echo "🚨 DRIFT FAIL-LOUD: requirements.txt uv.lock ile senkron değil! 'make requirements' çalıştırıp commitleyin." && exit 1)